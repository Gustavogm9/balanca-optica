"""Worker: processa vídeos de celular enviados pelo Pasto Leve (Supabase) e grava o peso.

Contrato (tabela public.pesagens_animais, bucket privado 'pesagens-midia'):
- pendente: fonte='celular' e metadados->>'status'='pendente', midia_path = '<fazenda_id>/<aaaa-mm>/<uuid>.<ext>'
- resultado: peso_estimado_kg, ic_inferior_kg, ic_superior_kg, n_frames, confiabilidade (0 a 1),
  versao_modelo e metadados mesclados com {status: 'processada'|'erro', reason, scale_src,
  marker_detected, processed_at, ...}.
- calibração por rebanho: pares (pesagem 'balanca', pesagem por câmera processada) do mesmo
  animal da mesma fazenda com |Δt| ≤ 2 dias; com ≥ 5 pares ajusta um HerdCalibrator.

Config por ambiente: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BO_MODEL, BO_SEG_WEIGHTS,
BO_POLL_SECONDS, BO_ONCE. Nunca registramos chaves em log.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..model.calibration import HerdCalibrator
from ..phone.estimate import estimate_from_media, load_model_bundle, reliability_score

log = logging.getLogger("bo.worker")

TABLE = "pesagens_animais"
BUCKET = "pesagens-midia"
MIN_PAIRS = 5
PAIR_WINDOW = timedelta(days=2)
CAMERA_SOURCES = ("celular", "camera_fixa")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_pairs(rows: list[dict], window: timedelta = PAIR_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Pares (peso da balança, peso da câmera) por animal: para cada pesagem de balança, a
    leitura por câmera processada mais próxima no tempo dentro da janela."""
    by_animal: dict[str, dict[str, list]] = {}
    for r in rows:
        if r.get("animal_id") is None or r.get("peso_estimado_kg") in (None, 0):
            continue
        d = by_animal.setdefault(str(r["animal_id"]), {"scale": [], "cam": []})
        ts = _parse_ts(r.get("medido_em") or r.get("created_at"))
        if ts is None:
            continue
        if r.get("fonte") == "balanca":
            d["scale"].append((ts, float(r["peso_estimado_kg"])))
        elif r.get("fonte") in CAMERA_SOURCES and (r.get("metadados") or {}).get("status") == "processada":
            d["cam"].append((ts, float(r["peso_estimado_kg"])))
    ys, ps = [], []
    for d in by_animal.values():
        for ts, y in d["scale"]:
            cands = [(abs(t - ts), p) for t, p in d["cam"] if abs(t - ts) <= window]
            if cands:
                ys.append(y); ps.append(min(cands)[1])
    return np.asarray(ys, float), np.asarray(ps, float)


@dataclass
class SupabaseStore:
    """Camada fina sobre o supabase-py, isolada para facilitar o mock nos testes."""
    client: object
    table: str = TABLE
    bucket: str = BUCKET

    def fetch_pending(self, limit: int = 10) -> list[dict]:
        res = (self.client.table(self.table).select("*").eq("fonte", "celular")
               .eq("metadados->>status", "pendente").order("created_at").limit(limit).execute())
        return list(res.data or [])

    def fetch_farm_rows(self, fazenda_id, since_days: int = 365) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        res = (self.client.table(self.table)
               .select("animal_id,fonte,peso_estimado_kg,medido_em,created_at,metadados")
               .eq("fazenda_id", fazenda_id).in_("fonte", ["balanca", *CAMERA_SOURCES])
               .not_.is_("peso_estimado_kg", "null").gte("created_at", since).execute())
        return list(res.data or [])

    def download(self, midia_path: str, dest_dir: Path) -> Path:
        data = self.client.storage.from_(self.bucket).download(midia_path)
        out = dest_dir / Path(midia_path).name
        out.write_bytes(data)
        return out

    def update(self, row_id, payload: dict) -> None:
        self.client.table(self.table).update(payload).eq("id", row_id).execute()


@dataclass
class Worker:
    store: SupabaseStore
    model: object
    segmenter: object
    ci: object | None = None
    model_version: str | None = None
    every_n: int = 6
    max_frames: int = 40
    _calib_cache: dict = field(default_factory=dict)

    def calibrator_for(self, fazenda_id) -> HerdCalibrator | None:
        if fazenda_id is None:
            return None
        key = str(fazenda_id)
        if key in self._calib_cache:
            return self._calib_cache[key]
        cal = None
        try:
            y, p = build_pairs(self.store.fetch_farm_rows(fazenda_id))
            if len(y) >= MIN_PAIRS:
                cal = HerdCalibrator(herd_id=key).fit(y, p)
                log.info("fazenda %s: calibração com %d pares (a=%+.4f b=%.4f)", key, cal.n_pairs, cal.a, cal.b)
            else:
                log.info("fazenda %s: %d pares (< %d), sem calibração", key, len(y), MIN_PAIRS)
        except Exception as e:  # noqa: BLE001
            log.warning("fazenda %s: falha ao montar calibração (%s); seguindo sem", key, e)
        self._calib_cache[key] = cal
        return cal

    def process_record(self, rec: dict) -> dict:
        """Processa um registro pendente e grava o resultado. Devolve o payload gravado."""
        rid = rec["id"]
        meta = dict(rec.get("metadados") or {})
        self.store.update(rid, {"metadados": {**meta, "status": "processando", "worker_started_at": _now_iso()}})
        try:
            if not rec.get("midia_path"):
                raise ValueError("midia_path vazio")
            with tempfile.TemporaryDirectory(prefix="bo_media_") as td:
                local = self.store.download(rec["midia_path"], Path(td))
                cal = self.calibrator_for(rec.get("fazenda_id"))
                res = estimate_from_media(local, self.model, self.segmenter, ci=self.ci, calibrator=cal,
                                          every_n=self.every_n, max_frames=self.max_frames,
                                          distance_m=meta.get("distance_m"), marker_m=meta.get("marker_m"),
                                          height_cm=meta.get("height_cm"),
                                          model_version=self.model_version)
            ok = res["weight_kg"] is not None
            payload = {
                "peso_estimado_kg": res["weight_kg"],
                "ic_inferior_kg": res["ci_low_kg"],
                "ic_superior_kg": res["ci_high_kg"],
                "n_frames": res["n_frames"],
                "confiabilidade": reliability_score(res),
                "versao_modelo": res["model_version"],
                "metadados": {**meta, "status": "processada" if ok else "erro",
                              "reason": res.get("reason"), "scale_src": res["scale_src"],
                              "marker_detected": res["marker_detected"], "reliable": res["reliable"],
                              "rel_mad": res["rel_mad"], "n_frames_total": res["n_frames_total"],
                              "calibration": res.get("calibration"), "processed_at": _now_iso()},
            }
        except Exception as e:  # noqa: BLE001
            log.error("registro %s: erro (%s)", rid, e)
            log.debug(traceback.format_exc())
            payload = {"metadados": {**meta, "status": "erro", "reason": f"{type(e).__name__}: {e}"[:500],
                                     "processed_at": _now_iso()}}
        self.store.update(rid, payload)
        return payload

    def run_once(self, limit: int = 10) -> int:
        self._calib_cache.clear()
        try:
            pending = self.store.fetch_pending(limit)
        except Exception as e:  # noqa: BLE001
            log.error("falha ao buscar pendentes: %s", e)
            return 0
        for rec in pending:
            try:
                self.process_record(rec)
            except Exception as e:  # noqa: BLE001  (o update final falhou; não derruba o loop)
                log.error("registro %s: falha irrecuperável (%s)", rec.get("id"), e)
        return len(pending)

    def run_forever(self, poll_seconds: float = 30.0, limit: int = 10) -> None:
        log.info("worker em loop (poll %.0fs, até %d por ciclo)", poll_seconds, limit)
        while True:
            n = self.run_once(limit)
            if n < limit:
                time.sleep(poll_seconds)


def make_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from supabase import create_client
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("defina SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY (ver .env.example)")
    return create_client(url, key)


def main(once: bool | None = None, limit: int = 10, poll_seconds: float | None = None) -> None:
    logging.basicConfig(level=os.environ.get("BO_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model_path = os.environ.get("BO_MODEL", "models/phone_v0_bai.joblib")
    weights = os.environ.get("BO_SEG_WEIGHTS", "yolov8n-seg.pt")
    poll = float(poll_seconds if poll_seconds is not None else os.environ.get("BO_POLL_SECONDS", "30"))
    if once is None:
        once = os.environ.get("BO_ONCE", "").lower() in ("1", "true", "yes")
    store = SupabaseStore(make_client())  # valida o ambiente antes de carregar modelos
    model, ci, version = load_model_bundle(model_path)
    from ..segmentation.yolo import YoloSegmenter
    seg = YoloSegmenter(weights, conf=0.3)
    w = Worker(store, model, seg, ci=ci, model_version=version)
    log.info("modelo %s (%s); segmentador %s", model_path, version, weights)
    if once:
        n = w.run_once(limit)
        log.info("%d registro(s) processado(s)", n)
    else:
        w.run_forever(poll, limit)
