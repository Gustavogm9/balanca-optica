"""Estimativa de peso a partir de uma mídia de celular (foto ou vídeo) já em disco.

Reaproveita `process_media` (um PhoneRecord sem peso), filtra os frames aprovados no
selo de qualidade, prediz por frame com o regressor de silhueta, agrega pela mediana,
aplica a calibração do rebanho (se houver) e o intervalo conformal. É o núcleo
compartilhado pelo worker do Supabase e pela API HTTP (`bo serve`).

Sem escala (nenhum frame com marcador, distância ou altura) a silhueta não tem
unidade física e o modelo não serve: devolvemos weight_kg=None e reason='sem_escala'.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..aggregate import aggregate_frames
from ..geometry.silhouette import SILHOUETTE_FEATURES
from ..model.calibration import HerdCalibrator
from ..model.regressor import WeightRegressor
from ..model.uncertainty import ConformalInterval
from .import_media import PhoneRecord, process_media
from .marker import DEFAULT_SIDE_M

SIL = list(SILHOUETTE_FEATURES)


def load_model_bundle(model_path: str | Path) -> tuple[WeightRegressor, ConformalInterval | None, str | None]:
    """Carrega o .joblib e o .ci.json ao lado (se existir). Devolve (modelo, ci, versão)."""
    model_path = Path(model_path)
    model = WeightRegressor.load(model_path)
    ci, version = None, getattr(model, "version", None)
    ci_path = Path(str(model_path) + ".ci.json")
    if ci_path.exists():
        d = json.loads(ci_path.read_text())
        ci = ConformalInterval(alpha=d["alpha"], q_rel=d["q_rel"])
        version = version or d.get("model_version")
    return model, ci, version or model_path.stem


def estimate_from_media(path: str | Path, model: WeightRegressor, segmenter,
                        ci: ConformalInterval | None = None,
                        calibrator: HerdCalibrator | None = None,
                        every_n: int = 6, max_frames: int = 40,
                        distance_m: float | None = None, marker_m: float | None = None,
                        height_cm: float | None = None, model_version: str | None = None) -> dict:
    """Uma leitura de peso para a mídia em `path`.

    distance_m / marker_m / height_cm são opcionais: marker_m é o lado do ArUco impresso
    (padrão 0,16 m); distance_m vem do LiDAR/ARKit; height_cm só para experimentos.
    """
    path = Path(path)
    version = model_version or getattr(model, "version", None) or "unknown"
    base = {"weight_kg": None, "ci_low_kg": None, "ci_high_kg": None, "n_frames": 0,
            "n_frames_total": 0, "rel_mad": None, "reliable": False, "scale_src": "none",
            "marker_detected": False, "model_version": version, "reason": None}
    rec = PhoneRecord(animal_id="?", weight_kg=None, file=path, view="lateral",
                      distance_m=distance_m, height_cm=height_cm,
                      aruco_side_m=marker_m or DEFAULT_SIDE_M)
    df = process_media([rec], segmenter, every_n=every_n, max_frames=max_frames, verbose=False)
    if df.empty:
        return {**base, "reason": "sem_animal"}
    base["n_frames_total"] = len(df)
    base["marker_detected"] = bool((df.scale_src == "aruco").any())
    scaled = df[df.scale_src != "none"]
    if scaled.empty:
        return {**base, "reason": "sem_escala"}
    base["scale_src"] = Counter(scaled.scale_src).most_common(1)[0][0]
    ok = scaled[scaled.q_ok] if "q_ok" in scaled else scaled
    if ok.empty:
        return {**base, "reason": "sem_frame_aprovado"}
    preds = model.predict(ok)  # o ColumnTransformer escolhe as colunas do modelo
    agg = aggregate_frames(np.asarray(preds, float))
    w = agg["weight"]
    if calibrator is not None and not np.isnan(w):
        w = float(calibrator.apply(np.array([w]))[0])
    out = {**base, "weight_kg": round(float(w), 1), "n_frames": agg["n_frames"],
           "rel_mad": round(float(agg["rel_mad"]), 4), "reliable": bool(agg["reliable"])}
    if ci is not None:
        lo, hi = ci.interval(np.array([w]), agg["n_frames"])
        out["ci_low_kg"], out["ci_high_kg"] = round(float(lo[0]), 1), round(float(hi[0]), 1)
    if calibrator is not None:
        out["calibration"] = {"herd_id": calibrator.herd_id, "n_pairs": calibrator.n_pairs}
    return out


def reliability_score(res: dict) -> float:
    """Nota 0 a 1 para a coluna `confiabilidade`: 1 = ≥3 frames aprovados e dispersão
    entre frames < 6%; cai linearmente com a dispersão (0 a 12%) e com menos frames."""
    if res.get("weight_kg") is None or not res.get("n_frames"):
        return 0.0
    mad = res.get("rel_mad") or 0.0
    disp = float(np.clip(1.0 - mad / 0.12, 0.0, 1.0))
    frames = min(res["n_frames"] / 3.0, 1.0)
    return round(disp * frames, 2)
