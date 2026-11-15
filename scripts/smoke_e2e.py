#!/usr/bin/env python
"""Smoke test ponta a ponta: Pasto Leve (Supabase) + motor Balança Óptica.

Roda contra um projeto Supabase REAL (de preferência de desenvolvimento) e percorre
o fluxo completo que o app e o worker executam em produção:

  1. cria um usuário de teste pela Admin API (service role) e faz login (anon key)
  2. cria fazenda, lote e animal via REST como esse usuário (RLS valendo)
  3. gera um vídeo sintético de 3 s com marcador ArUco e um "corpo" escuro
  4. sobe o vídeo no bucket privado `pesagens-midia` com o JWT do usuário
  5. insere a pesagem pendente (fonte celular, metadados.status = pendente)
  6. chama a Edge Function `ingestao-pesagem` com JWT (peso de balança) e com
     X-Device-Token (dispositivo criado antes via service role + RPC hash_device_token)
  7. roda o worker em processo (`supabase_worker.main(once=True)`)
  8. confere o registro processado, a view vw_gmd_animal e o trigger de peso
  9. limpa tudo (registros, arquivo, dispositivo, usuário), a menos que --keep

Variáveis obrigatórias no ambiente (ou em um .env na raiz do repositório):
  SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
Opcionais: BO_MODEL, BO_SEG_WEIGHTS (padrão: arquivos da raiz do repositório).

Sem as variáveis o script imprime instruções e sai com código 2. Saída: checklist
PASS/FAIL por etapa; código 0 quando tudo passou, 1 quando algo falhou.

As funções puras (vídeo sintético, payloads, avaliação do resultado) não tocam a
rede e são testadas em tests/test_smoke_script.py.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import string
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "pesagens-midia"
TABLE = "pesagens_animais"
EDGE_FUNCTION = "ingestao-pesagem"
REQUIRED_ENV = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")

# Razões de erro do worker aceitas como sucesso do FLUXO: o vídeo é sintético e o
# YOLO (classe 'cow' do COCO) pode legitimamente não ver um boi no retângulo escuro.
ACCEPTED_ERROR_REASONS = frozenset({"sem_animal"})

PESO_BALANCA_KG = 412.5
PESO_DISPOSITIVO_KG = 420.0


# ----------------------------------------------------------------------------- funções puras

def smoke_email(ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else int(ts)
    return f"smoke+{ts}@guilds.test"


def random_password(n: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#%*-_"
    pw = "".join(secrets.choice(alphabet) for _ in range(n))
    # garante as classes que o GoTrue costuma exigir
    return "Aa1!" + pw[4:]


def midia_path_for(fazenda_id: str, media_id: str | None = None, ext: str = "mp4") -> str:
    media_id = media_id or str(uuid.uuid4())
    return f"{fazenda_id}/smoke/{media_id}.{ext}"


def make_synthetic_video(path: str | Path, seconds: float = 3.0, fps: int = 30,
                         size: tuple[int, int] = (960, 720), marker_id: int = 5,
                         seed: int = 0) -> Path:
    """Vídeo mp4 com fundo claro ruidoso, um 'corpo' escuro (tronco + perna) ocupando
    o centro e um marcador ArUco (4x4_50) no canto superior direito, em todos os frames.
    Estrutura igual à do tests/test_worker.py para o detector de marcador funcionar."""
    import cv2

    from balanca_optica.phone.marker import generate_marker

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    n = max(1, round(seconds * fps))
    rng = np.random.default_rng(seed)
    # generate_marker devolve a folha inteira (marcador + régua + texto); usamos só o
    # quadrado com borda branca e o redimensionamos para ~1/3 da altura do frame
    sheet = generate_marker(marker_id, side_px=240, border_px=30)
    square = sheet[:300, :300]
    side = max(60, h // 4)
    marker = cv2.resize(square, (side, side), interpolation=cv2.INTER_NEAREST)
    mh, mw = marker.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter não abriu (codec mp4v indisponível?)")
    try:
        for _ in range(n):
            img = cv2.add(np.full((h, w), 150, np.uint8), rng.integers(0, 40, size=(h, w), dtype=np.uint8))
            # corpo escuro: tronco retangular com cantos suaves + uma perna
            cv2.rectangle(img, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), 35, -1)
            cv2.rectangle(img, (w // 4, 2 * h // 3), (w // 4 + w // 20, int(0.9 * h)), 35, -1)
            cv2.ellipse(img, (3 * w // 4, h // 2), (w // 16, h // 6), 0, 0, 360, 35, -1)  # "cabeça"
            # marcador no canto superior direito, sem sobrepor o corpo
            y0, x0 = 40, w - mw - 40
            img[y0:y0 + mh, x0:x0 + mw] = marker
            writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    return path


def build_fazenda(user_id: str, ts: int) -> dict:
    return {"user_id": user_id, "nome": f"Fazenda Smoke {ts}", "proprietario": "Smoke Test",
            "municipio": "São José do Rio Preto", "uf": "SP", "area_ha": 10.5}


def build_lote(fazenda_id: str) -> dict:
    return {"fazenda_id": fazenda_id, "nome": "Lote Smoke", "categoria": "boi",
            "quantidade": 1, "peso_medio_kg": 0, "observacao": "criado pelo smoke_e2e"}


def build_animal(fazenda_id: str, lote_id: str, ts: int) -> dict:
    return {"fazenda_id": fazenda_id, "identificador": f"SMOKE-{ts}", "nome": "Boi Smoke",
            "categoria": "boi", "lote_id": lote_id, "raca": "Nelore", "peso_kg": 400}


def build_pesagem_pendente(fazenda_id: str, animal_id: str, lote_id: str, midia_path: str,
                           medido_em: str | None = None, marker_id: int = 5) -> dict:
    """Mesmo registro que o PWA insere em src/lib/pesagemCameraApi.ts (peso 0 até o worker)."""
    medido_em = medido_em or datetime.now(timezone.utc).isoformat()
    return {
        "fazenda_id": fazenda_id, "animal_id": animal_id, "lote_id": lote_id,
        "peso_estimado_kg": 0, "ic_inferior_kg": None, "ic_superior_kg": None,
        "n_frames": None, "confiabilidade": None, "fonte": "celular", "medido_em": medido_em,
        "dispositivo_id": "smoke-e2e", "versao_modelo": None, "midia_path": midia_path,
        "metadados": {"status": "pendente", "midia_path": midia_path, "marcador_detectado": True,
                      "marcador_id": marker_id, "origem": "smoke_e2e", "observacoes": None,
                      "peso_balanca_kg": None},
    }


def build_ingestao_jwt(fazenda_id: str, animal_id: str, lote_id: str,
                       peso_kg: float = PESO_BALANCA_KG, medido_em: str | None = None) -> dict:
    """Pesagem de balança digitada à mão, enviada pelo usuário logado (exige fazenda_id)."""
    medido_em = medido_em or (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    return {"fazenda_id": fazenda_id, "animal_id": animal_id, "lote_id": lote_id,
            "peso_estimado_kg": float(peso_kg), "fonte": "balanca", "medido_em": medido_em,
            "metadados": {"origem": "smoke_e2e", "entrada": "manual"}}


def build_ingestao_device(animal_id: str, peso_kg: float = PESO_DISPOSITIVO_KG,
                          medido_em: str | None = None) -> dict:
    """Leitura de câmera fixa: sem fazenda_id (vem do dispositivo), confiável (>= 0,7)."""
    medido_em = medido_em or datetime.now(timezone.utc).isoformat()
    return {"animal_id": animal_id, "peso_estimado_kg": float(peso_kg), "ic_inferior_kg": peso_kg - 15,
            "ic_superior_kg": peso_kg + 15, "n_frames": 12, "confiabilidade": 0.9,
            "fonte": "camera_fixa", "medido_em": medido_em, "versao_modelo": "smoke_e2e",
            "metadados": {"origem": "smoke_e2e"}}


def build_dispositivo(fazenda_id: str, token_hash: str, ts: int) -> dict:
    return {"fazenda_id": fazenda_id, "nome": f"Câmera smoke {ts}", "token_hash": token_hash, "ativo": True}


def expected_animal_weight(device_ok: bool, jwt_ok: bool) -> float | None:
    """Peso que o trigger deve ter deixado em animais.peso_kg: a última inserção confiável
    vence (dispositivo depois da balança). None quando nenhuma ingestão foi gravada."""
    if device_ok:
        return PESO_DISPOSITIVO_KG
    if jwt_ok:
        return PESO_BALANCA_KG
    return None


def evaluate_processed(row: dict) -> tuple[bool, str]:
    """(ok, descrição) para o registro depois do worker. 'processada' passa; 'erro' passa
    só com reason em ACCEPTED_ERROR_REASONS; qualquer outro estado falha."""
    meta = row.get("metadados") or {}
    status = meta.get("status")
    if status == "processada":
        peso = row.get("peso_estimado_kg")
        if peso is None or float(peso) <= 0:
            return False, "status processada mas peso_estimado_kg vazio/zero"
        return True, (f"processada: {float(peso):.1f} kg, IC [{row.get('ic_inferior_kg')}, "
                      f"{row.get('ic_superior_kg')}], {row.get('n_frames')} frames, "
                      f"escala={meta.get('scale_src')}, marcador={meta.get('marker_detected')}, "
                      f"confiabilidade={row.get('confiabilidade')}, modelo={row.get('versao_modelo')}")
    if status == "erro":
        reason = str(meta.get("reason") or "")
        if reason in ACCEPTED_ERROR_REASONS:
            return True, f"erro aceito ({reason}): vídeo sintético, YOLO não viu boi; fluxo completo"
        return False, f"erro inesperado do worker: {reason or '(sem reason)'}"
    if status in ("pendente", "processando"):
        return False, f"worker não concluiu: status ainda '{status}'"
    return False, f"status desconhecido: {status!r}"


# ----------------------------------------------------------------------------- checklist

@dataclass
class Step:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


@dataclass
class Checklist:
    steps: list[Step] = field(default_factory=list)
    verbose: bool = False

    def run(self, name: str, fn, critical: bool = True):
        """Executa fn(); registra PASS/FAIL. Falha em etapa crítica aborta (StepAborted)."""
        try:
            detail = fn()
            self.steps.append(Step(name, "PASS", str(detail or "")))
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""), flush=True)
            return detail
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:400]
            self.steps.append(Step(name, "FAIL", msg))
            print(f"  FAIL  {name}  -> {msg}", flush=True)
            if self.verbose:
                traceback.print_exc()
            if critical:
                raise StepAborted(name) from e
            return None

    def skip(self, name: str, why: str = "etapa anterior falhou"):
        self.steps.append(Step(name, "SKIP", why))
        print(f"  SKIP  {name}  ({why})", flush=True)

    @property
    def failed(self) -> list[Step]:
        return [s for s in self.steps if s.status == "FAIL"]

    def summary(self) -> str:
        n_pass = sum(s.status == "PASS" for s in self.steps)
        n_fail = len(self.failed)
        n_skip = sum(s.status == "SKIP" for s in self.steps)
        lines = ["", "=" * 72, "RESUMO"]
        for s in self.steps:
            lines.append(f"  [{s.status:4}] {s.name}")
        lines.append(f"  {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
        return "\n".join(lines)


class StepAborted(RuntimeError):
    pass


# ----------------------------------------------------------------------------- fluxo real

def load_env() -> dict[str, str] | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    env = {k: os.environ.get(k, "") for k in REQUIRED_ENV}
    missing = [k for k, v in env.items() if not v]
    if missing:
        print("smoke_e2e: faltam variáveis de ambiente:", ", ".join(missing))
        print()
        print("Exporte (ou coloque em um .env na raiz do repositório):")
        print("  SUPABASE_URL=https://<ref>.supabase.co")
        print("  SUPABASE_ANON_KEY=<anon/publishable key>           (Dashboard > Settings > API)")
        print("  SUPABASE_SERVICE_ROLE_KEY=<service_role key>       (nunca no app; só aqui e no worker)")
        print()
        print("Pré-requisitos no projeto Supabase: migrations do Pasto Leve aplicadas")
        print("(tabelas pesagens_animais/dispositivos, RPC hash_device_token, view vw_gmd_animal,")
        print("bucket pesagens-midia) e a Edge Function ingestao-pesagem publicada.")
        print("Dependências Python: pip install -e '.[worker,yolo,gbm]' requests")
        return None
    return env


def _rest_headers(anon: str, jwt: str) -> dict:
    return {"apikey": anon, "Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}


def _call_edge(url: str, anon: str, body: dict, *, jwt: str | None = None,
               device_token: str | None = None) -> tuple[int, dict]:
    import requests
    headers = {"apikey": anon, "Content-Type": "application/json"}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"
    if device_token:
        headers["X-Device-Token"] = device_token
    r = requests.post(f"{url.rstrip('/')}/functions/v1/{EDGE_FUNCTION}", headers=headers,
                      data=json.dumps(body), timeout=60)
    try:
        payload = r.json()
    except ValueError:
        payload = {"raw": r.text[:500]}
    return r.status_code, payload


def _single(res) -> dict:
    data = res.data
    if isinstance(data, list):
        if not data:
            raise RuntimeError("resposta vazia (RLS bloqueou ou registro não existe)")
        return data[0]
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Smoke test ponta a ponta Pasto Leve + Balança Óptica")
    ap.add_argument("--keep", action="store_true", help="não apaga usuário/registros/arquivo no fim")
    ap.add_argument("--worker-limit", type=int, default=10, help="registros por ciclo do worker")
    ap.add_argument("--seconds", type=float, default=3.0, help="duração do vídeo sintético")
    ap.add_argument("-v", "--verbose", action="store_true", help="traceback completo nas falhas")
    a = ap.parse_args(argv)

    env = load_env()
    if env is None:
        return 2
    url, anon, service = env["SUPABASE_URL"], env["SUPABASE_ANON_KEY"], env["SUPABASE_SERVICE_ROLE_KEY"]
    os.environ.setdefault("BO_MODEL", str(REPO_ROOT / "models" / "phone_v0_bai.joblib"))
    os.environ.setdefault("BO_SEG_WEIGHTS", str(REPO_ROOT / "yolov8n-seg.pt"))

    from supabase import create_client

    ts = int(time.time())
    email, password = smoke_email(ts), random_password()
    ck = Checklist(verbose=a.verbose)
    ctx: dict = {"ts": ts, "email": email}
    print(f"smoke_e2e contra {url} (usuário {email})")
    print("-" * 72)

    admin = create_client(url, service)
    user_client = None

    try:
        def s_create_user():
            res = admin.auth.admin.create_user({"email": email, "password": password, "email_confirm": True,
                                                "user_metadata": {"nome": "Smoke Test"}})
            ctx["user_id"] = res.user.id
            return res.user.id
        ck.run("1. criar usuário de teste (admin API)", s_create_user)

        def s_login():
            nonlocal user_client
            user_client = create_client(url, anon)
            res = user_client.auth.sign_in_with_password({"email": email, "password": password})
            if not res.session or not res.session.access_token:
                raise RuntimeError("login sem sessão")
            ctx["jwt"] = res.session.access_token
            return f"JWT com {len(ctx['jwt'])} chars"
        ck.run("2. login com anon key (JWT do usuário)", s_login)

        def s_fazenda():
            row = _single(user_client.table("fazendas").insert(build_fazenda(ctx["user_id"], ts)).execute())
            ctx["fazenda_id"] = row["id"]
            return row["id"]
        ck.run("3. criar fazenda via REST (RLS)", s_fazenda)

        def s_lote():
            row = _single(user_client.table("lotes").insert(build_lote(ctx["fazenda_id"])).execute())
            ctx["lote_id"] = row["id"]
            return row["id"]
        ck.run("4. criar lote via REST (RLS)", s_lote)

        def s_animal():
            row = _single(user_client.table("animais").insert(
                build_animal(ctx["fazenda_id"], ctx["lote_id"], ts)).execute())
            ctx["animal_id"] = row["id"]
            return row["id"]
        ck.run("5. criar animal via REST (RLS)", s_animal)

        def s_video():
            import tempfile
            td = Path(tempfile.mkdtemp(prefix="bo_smoke_"))
            ctx["tmpdir"] = td
            p = make_synthetic_video(td / "smoke.mp4", seconds=a.seconds)
            size = p.stat().st_size
            if size < 1000:
                raise RuntimeError(f"vídeo muito pequeno ({size} bytes)")
            ctx["video"] = p
            return f"{size / 1024:.0f} KB"
        ck.run("6. gerar vídeo sintético (ArUco + corpo escuro)", s_video)

        def s_upload():
            path = midia_path_for(ctx["fazenda_id"])
            user_client.storage.from_(BUCKET).upload(path, ctx["video"].read_bytes(),
                                                     {"content-type": "video/mp4", "upsert": "false"})
            ctx["midia_path"] = path
            return path
        ck.run("7. upload no bucket pesagens-midia com JWT do usuário", s_upload)

        def s_pendente():
            payload = build_pesagem_pendente(ctx["fazenda_id"], ctx["animal_id"], ctx["lote_id"], ctx["midia_path"])
            row = _single(user_client.table(TABLE).insert(payload).execute())
            ctx["pendente_id"] = row["id"]
            return row["id"]
        ck.run("8. inserir pesagem pendente (fonte celular)", s_pendente)

        def s_edge_jwt():
            status, body = _call_edge(url, anon, build_ingestao_jwt(ctx["fazenda_id"], ctx["animal_id"], ctx["lote_id"]),
                                      jwt=ctx["jwt"])
            if status != 201 or not body.get("id"):
                raise RuntimeError(f"HTTP {status}: {body}")
            ctx["ingest_jwt_id"] = body["id"]
            return f"id {body['id']}"
        ck.run("9. Edge Function ingestao-pesagem com JWT (peso balança)", s_edge_jwt, critical=False)

        def s_device():
            token = secrets.token_hex(32)
            h = admin.rpc("hash_device_token", {"token": token}).execute().data
            if not isinstance(h, str) or len(h) != 64:
                raise RuntimeError(f"hash_device_token devolveu {h!r}")
            row = _single(admin.table("dispositivos").insert(build_dispositivo(ctx["fazenda_id"], h, ts)).execute())
            ctx["device_id"], ctx["device_token"] = row["id"], token
            return row["id"]
        ck.run("10. criar dispositivo via service role + RPC hash_device_token", s_device, critical=False)

        if "device_token" in ctx:
            def s_edge_device():
                status, body = _call_edge(url, anon, build_ingestao_device(ctx["animal_id"]),
                                          device_token=ctx["device_token"])
                if status != 201 or not body.get("id"):
                    raise RuntimeError(f"HTTP {status}: {body}")
                row = _single(admin.table(TABLE).select("fazenda_id,metadados,dispositivo_id")
                              .eq("id", body["id"]).execute())
                if row["fazenda_id"] != ctx["fazenda_id"]:
                    raise RuntimeError("fazenda_id gravada difere da fazenda do dispositivo")
                if (row.get("metadados") or {}).get("dispositivo_registro_id") != ctx["device_id"]:
                    raise RuntimeError("metadados.dispositivo_registro_id não aponta para o dispositivo")
                ctx["ingest_dev_id"] = body["id"]
                return f"id {body['id']}"
            ck.run("11. Edge Function ingestao-pesagem com X-Device-Token", s_edge_device, critical=False)
        else:
            ck.skip("11. Edge Function ingestao-pesagem com X-Device-Token")

        def s_worker():
            from balanca_optica.worker import supabase_worker
            t0 = time.time()
            supabase_worker.main(once=True, limit=a.worker_limit)
            return f"{time.time() - t0:.1f} s"
        ck.run("12. worker em processo (supabase_worker.main(once=True))", s_worker, critical=False)

        def s_check_row():
            row = _single(admin.table(TABLE).select("*").eq("id", ctx["pendente_id"]).execute())
            ok, detail = evaluate_processed(row)
            if not ok:
                raise RuntimeError(detail)
            return detail
        ck.run("13. registro pendente virou processada/erro aceito", s_check_row, critical=False)

        def s_gmd():
            rows = user_client.table("vw_gmd_animal").select("*").eq("animal_id", ctx["animal_id"]).execute().data
            if not rows:
                raise RuntimeError("view vazia para o animal (RLS ou sem pesagens com animal_id)")
            r = rows[0]
            return (f"n_pesagens={r.get('n_pesagens')} primeiro={r.get('primeiro_peso_kg')} "
                    f"ultimo={r.get('ultimo_peso_kg')} gmd={r.get('gmd_kg_dia')} kg/dia")
        ck.run("14. view vw_gmd_animal consultável como usuário", s_gmd, critical=False)

        esperado = expected_animal_weight("ingest_dev_id" in ctx, "ingest_jwt_id" in ctx)
        if esperado is None:
            ck.skip("15. trigger atualizou animais.peso_kg", "nenhuma ingestão confiável foi gravada")
        else:
            def s_trigger():
                row = _single(user_client.table("animais").select("peso_kg").eq("id", ctx["animal_id"]).execute())
                peso = float(row["peso_kg"] or 0)
                if abs(peso - esperado) > 0.01:
                    raise RuntimeError(f"animais.peso_kg = {peso}, esperado {esperado} (trigger trg_aplicar_pesagem_animal)")
                return f"animais.peso_kg = {peso}"
            ck.run("15. trigger atualizou animais.peso_kg", s_trigger, critical=False)

    except StepAborted as e:
        print(f"\nabortado em: {e}")
    except KeyboardInterrupt:
        print("\ninterrompido")
    finally:
        if a.keep:
            print("\n--keep: nada foi apagado. IDs:")
            print(json.dumps({k: str(v) for k, v in ctx.items() if k not in ("jwt", "device_token", "video", "tmpdir")},
                             indent=2, ensure_ascii=False))
        else:
            def s_cleanup():
                errs = []
                def attempt(label, fn):
                    try:
                        fn()
                    except Exception as e:  # noqa: BLE001
                        errs.append(f"{label}: {type(e).__name__}: {e}"[:160])
                if "midia_path" in ctx:
                    attempt("storage", lambda: admin.storage.from_(BUCKET).remove([ctx["midia_path"]]))
                if "fazenda_id" in ctx:
                    fz = ctx["fazenda_id"]
                    attempt(TABLE, lambda: admin.table(TABLE).delete().eq("fazenda_id", fz).execute())
                    attempt("dispositivos", lambda: admin.table("dispositivos").delete().eq("fazenda_id", fz).execute())
                    attempt("animais", lambda: admin.table("animais").delete().eq("fazenda_id", fz).execute())
                    attempt("lotes", lambda: admin.table("lotes").delete().eq("fazenda_id", fz).execute())
                    attempt("fazendas", lambda: admin.table("fazendas").delete().eq("id", fz).execute())
                if "user_id" in ctx:
                    attempt("auth.user", lambda: admin.auth.admin.delete_user(ctx["user_id"]))
                if "tmpdir" in ctx:
                    import shutil
                    attempt("tmpdir", lambda: shutil.rmtree(ctx["tmpdir"], ignore_errors=True))
                if errs:
                    raise RuntimeError("; ".join(errs))
                return "registros, arquivo, dispositivo e usuário removidos"
            print()
            ck.run("16. limpeza", s_cleanup, critical=False)
        try:
            if user_client is not None:
                user_client.auth.sign_out()
        except Exception as e:  # noqa: BLE001
            print(f"  (aviso) sign_out falhou: {e}")

    print(ck.summary())
    return 1 if ck.failed else 0


if __name__ == "__main__":
    sys.exit(main())
