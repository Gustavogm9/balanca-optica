"""estimate_from_media (vídeo sintético com ArUco), API HTTP e worker com Supabase mockado."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from balanca_optica.geometry.silhouette import SILHOUETTE_FEATURES, silhouette_features
from balanca_optica.model.regressor import WeightRegressor
from balanca_optica.model.uncertainty import ConformalInterval
from balanca_optica.phone.estimate import estimate_from_media, reliability_score
from balanca_optica.phone.marker import generate_marker
from balanca_optica.worker import supabase_worker as sw

W, H = 960, 720


class _FakeSeg:
    """Mesmo 'animal' do test_capture_phone: tronco retangular + uma perna."""
    def mask(self, img):
        m = np.zeros(img.shape[:2], bool); h, w = m.shape
        m[h // 3:2 * h // 3, w // 4:3 * w // 4] = True
        m[2 * h // 3:int(0.9 * h), w // 4:w // 4 + w // 20] = True
        return m


def _write_video(path: Path, with_marker: bool, n: int = 36) -> Path:
    rng = np.random.default_rng(0)
    marker = generate_marker(5, side_px=240, border_px=30)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    assert wr.isOpened()
    for _ in range(n):
        img = cv2.add(np.full((H, W), 120, np.uint8), rng.integers(0, 40, size=(H, W), dtype=np.uint8))
        if with_marker:
            img[80:80 + marker.shape[0], 640:640 + marker.shape[1]] = marker
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    wr.release()
    return path


def _synthetic_regressor(seed: int = 0):
    """Regressor de silhueta treinado em máscaras sintéticas de vários tamanhos/escala,
    com peso = 320·área_do_tronco^1.4 (alométrico), para o teste ser determinístico."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for _ in range(60):
        m = np.zeros((300, 400), bool)
        bh, bw = int(rng.integers(60, 140)), int(rng.integers(150, 300))
        r0, c0 = (300 - bh) // 2, (400 - bw) // 2
        m[r0:r0 + bh, c0:c0 + bw] = True
        m[r0 + bh:min(r0 + bh + 40, 299), c0:c0 + 12] = True
        px_per_m = float(rng.uniform(80, 200))
        f = silhouette_features(m, px_per_m)
        rows.append(f); ys.append(320.0 * f["sil_body_area"] ** 1.4 * rng.uniform(0.97, 1.03))
    df = pd.DataFrame(rows)
    m = WeightRegressor(kind="linear", numeric=list(SILHOUETTE_FEATURES), version="test_v0").fit(df, np.array(ys))
    ci = ConformalInterval(alpha=0.1).fit(np.array(ys), m.predict(df))
    return m, ci


@pytest.fixture(scope="module")
def regressor():
    return _synthetic_regressor()


def test_estimate_from_video_with_aruco(tmp_path, regressor):
    model, ci = regressor
    video = _write_video(tmp_path / "v.mp4", with_marker=True)
    res = estimate_from_media(video, model, _FakeSeg(), ci=ci, every_n=6, max_frames=40)
    assert set(res) >= {"weight_kg", "ci_low_kg", "ci_high_kg", "n_frames", "n_frames_total",
                        "rel_mad", "reliable", "scale_src", "marker_detected", "model_version"}
    assert res["scale_src"] == "aruco" and res["marker_detected"] is True
    assert res["n_frames_total"] == 6 and res["n_frames"] == 6
    assert res["weight_kg"] is not None and res["weight_kg"] > 0
    assert res["ci_low_kg"] < res["weight_kg"] < res["ci_high_kg"]
    assert res["reliable"] is True and res["rel_mad"] < 0.02  # mesmo animal em todos os frames
    assert res["model_version"] == "test_v0" and res["reason"] is None
    # tronco 480x240 px a 1500 px/m -> 0,32 x 0,16 m: peso coerente com a lei de treino
    area = 0.32 * 0.16
    assert abs(res["weight_kg"] - 320 * area ** 1.4) / (320 * area ** 1.4) < 0.15
    assert 0.9 <= reliability_score(res) <= 1.0


def test_estimate_without_scale_returns_none(tmp_path, regressor):
    model, ci = regressor
    video = _write_video(tmp_path / "s.mp4", with_marker=False, n=12)
    res = estimate_from_media(video, model, _FakeSeg(), ci=ci)
    assert res["weight_kg"] is None and res["reason"] == "sem_escala"
    assert res["marker_detected"] is False and res["scale_src"] == "none"
    assert res["n_frames_total"] == 2 and reliability_score(res) == 0.0


def test_calibrator_is_applied(tmp_path, regressor):
    from balanca_optica.model.calibration import HerdCalibrator
    model, _ = regressor
    video = _write_video(tmp_path / "c.mp4", with_marker=True, n=12)
    raw = estimate_from_media(video, model, _FakeSeg())
    cal = HerdCalibrator(herd_id="f1", a=np.log(1.10), b=1.0, n_pairs=12)
    res = estimate_from_media(video, model, _FakeSeg(), calibrator=cal)
    assert abs(res["weight_kg"] / raw["weight_kg"] - 1.10) < 0.01
    assert res["calibration"] == {"herd_id": "f1", "n_pairs": 12}


def test_http_api(tmp_path, regressor):
    from fastapi.testclient import TestClient

    from balanca_optica.serve import create_app
    model, ci = regressor
    app = create_app(model=model, segmenter=_FakeSeg(), ci=ci, model_version="test_v0")
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok", "model_version": "test_v0"}
    video = _write_video(tmp_path / "api.mp4", with_marker=True, n=12)
    r = c.post("/estimate", files={"file": ("api.mp4", video.read_bytes(), "video/mp4")}, data={"every_n": "6"})
    assert r.status_code == 200
    body = r.json()
    assert body["scale_src"] == "aruco" and body["weight_kg"] > 0 and "confiabilidade" in body
    assert c.post("/estimate", files={"file": ("x.txt", b"oi", "text/plain")}).status_code == 415


# ---------------------------------------------------------------- worker com Supabase mockado

class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """Builder encadeável que registra filtros e devolve dados do _FakeClient."""
    def __init__(self, client, table):
        self.client, self.table, self.filters, self.op, self.payload = client, table, [], "select", None
        self.not_ = self  # `.not_.is_(...)`

    def select(self, *_): return self
    def order(self, *_, **__): return self
    def limit(self, *_): return self
    def eq(self, col, val): self.filters.append(("eq", col, val)); return self
    def in_(self, col, vals): self.filters.append(("in", col, vals)); return self
    def is_(self, col, val): self.filters.append(("is", col, val)); return self
    def gte(self, col, val): self.filters.append(("gte", col, val)); return self
    def update(self, payload): self.op, self.payload = "update", payload; return self

    def execute(self):
        if self.op == "update":
            rid = next(v for k, c, v in self.filters if c == "id")
            self.client.updates.append((rid, self.payload))
            for r in self.client.rows:
                if r["id"] == rid:
                    r.update(self.payload)
            return _Result([])
        rows = list(self.client.rows)
        for kind, col, val in self.filters:
            if kind == "eq" and col == "metadados->>status":
                rows = [r for r in rows if (r.get("metadados") or {}).get("status") == val]
            elif kind == "eq":
                rows = [r for r in rows if r.get(col) == val]
            elif kind == "in":
                rows = [r for r in rows if r.get(col) in val]
            elif kind == "is" and val == "null":  # usado só via .not_ -> peso não nulo
                rows = [r for r in rows if r.get(col) is not None]
        return _Result(rows)


class _Bucket:
    def __init__(self, files): self.files = files
    def download(self, path): return self.files[path]


class _Storage:
    def __init__(self, files): self.files = files
    def from_(self, bucket): assert bucket == "pesagens-midia"; return _Bucket(self.files)


class _FakeClient:
    def __init__(self, rows, files):
        self.rows, self.updates, self.storage = rows, [], _Storage(files)
    def table(self, name): assert name == "pesagens_animais"; return _Query(self, name)


def _farm_rows(fazenda="faz1", n_animals=8):
    """Pares balança/câmera históricos para a calibração: câmera sempre 10% abaixo da balança."""
    rows = []
    for i in range(n_animals):
        rows.append({"id": f"b{i}", "fazenda_id": fazenda, "animal_id": f"a{i}", "fonte": "balanca",
                     "peso_estimado_kg": 400 + 10 * i, "medido_em": "2026-11-01T10:00:00+00:00",
                     "metadados": {}, "midia_path": None})
        rows.append({"id": f"c{i}", "fazenda_id": fazenda, "animal_id": f"a{i}", "fonte": "celular",
                     "peso_estimado_kg": (400 + 10 * i) * 0.9, "medido_em": "2026-11-02T09:00:00+00:00",
                     "metadados": {"status": "processada"}, "midia_path": f"{fazenda}/2026-11/c{i}.mp4"})
    return rows


def test_build_pairs_window():
    rows = _farm_rows()
    rows.append({"animal_id": "a0", "fonte": "celular", "peso_estimado_kg": 999, "fazenda_id": "faz1",
                 "medido_em": "2026-10-01T10:00:00+00:00", "metadados": {"status": "processada"}})  # fora da janela
    y, p = sw.build_pairs(rows)
    assert len(y) == 8 and np.allclose(p / y, 0.9)


def test_worker_processes_pending_record(tmp_path, regressor, monkeypatch):
    model, ci = regressor
    video = _write_video(tmp_path / "p.mp4", with_marker=True, n=12)
    rows = _farm_rows() + [
        {"id": "p1", "fazenda_id": "faz1", "animal_id": "a99", "lote_id": None, "fonte": "celular",
         "medido_em": "2026-11-14T12:00:00+00:00", "midia_path": "faz1/2026-11/p1.mp4",
         "metadados": {"status": "pendente", "app": "pasto-leve"}, "peso_estimado_kg": None,
         "created_at": "2026-11-14T12:00:01+00:00"},
        {"id": "p2", "fazenda_id": "faz2", "animal_id": "z1", "fonte": "celular", "midia_path": "faz2/2026-11/p2.mp4",
         "metadados": {"status": "pendente"}, "peso_estimado_kg": None, "created_at": "2026-11-14T12:00:02+00:00"},
        {"id": "done", "fazenda_id": "faz1", "fonte": "celular", "midia_path": "x", "metadados": {"status": "processada"},
         "peso_estimado_kg": 500, "created_at": "2026-11-13T12:00:02+00:00"},
    ]
    client = _FakeClient(rows, {"faz1/2026-11/p1.mp4": video.read_bytes()})  # p2 sem arquivo -> erro
    monkeypatch.setattr(sw, "make_client", lambda: client)
    raw = estimate_from_media(video, model, _FakeSeg(), ci=ci)

    w = sw.Worker(sw.SupabaseStore(sw.make_client()), model, _FakeSeg(), ci=ci, model_version="test_v0")
    assert w.run_once(limit=10) == 2

    final = {rid: pl for rid, pl in client.updates}  # último update por id
    assert [rid for rid, _ in client.updates][:2] == ["p1", "p1"]  # claim ('processando') e resultado
    assert client.updates[0][1]["metadados"]["status"] == "processando"

    p1 = final["p1"]
    assert p1["metadados"]["status"] == "processada" and p1["metadados"]["app"] == "pasto-leve"
    assert p1["metadados"]["scale_src"] == "aruco" and p1["metadados"]["marker_detected"] is True
    assert p1["versao_modelo"] == "test_v0" and p1["n_frames"] == 2
    assert p1["ic_inferior_kg"] < p1["peso_estimado_kg"] < p1["ic_superior_kg"]
    assert 0 < p1["confiabilidade"] <= 1
    # calibração da faz1: 8 pares com câmera 10% abaixo -> corrige para cima (encolhido pelo prior)
    assert p1["metadados"]["calibration"]["herd_id"] == "faz1" and p1["metadados"]["calibration"]["n_pairs"] == 8
    assert 1.03 < p1["peso_estimado_kg"] / raw["weight_kg"] < 1.12
    assert "processed_at" in p1["metadados"]
    json.dumps(p1)  # serializável para o PostgREST

    p2 = final["p2"]
    assert p2["metadados"]["status"] == "erro" and "KeyError" in p2["metadados"]["reason"]
    assert "peso_estimado_kg" not in p2
    # registro já processado não foi tocado; fila vazia na segunda rodada
    assert all(rid != "done" for rid, _ in client.updates)
    assert w.run_once(limit=10) == 0
