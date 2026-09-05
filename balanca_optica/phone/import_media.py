"""Importador de fotos e vídeos de celular coletados no campo.

Entrada: um CSV de coleta (gerado pelo formulário `tools/coleta.html` ou à mão) com
colunas: animal_id, weight_kg, file, view (lateral|dorsal|traseira|outro),
[distance_m] [marker_px] [marker_m] [height_cm] [farm] [breed] [sex] [category]
e a pasta com as mídias. Saída: um CSV de features por frame, pronto para
`bo validate` / `bo train` (uma linha por frame; agregação por animal na avaliação).

Escala (px/m), na ordem de preferência: distance_m + focal (LiDAR/ARKit);
marker_px/marker_m (marcador impresso); height_cm (só experimentos). Sem nada, as
features saem normalizadas pela diagonal da imagem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..geometry.silhouette import (scale_from_distance, scale_from_known_height,
                                   scale_from_marker, silhouette_features)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
MAX_SIDE = 1280  # redimensiona para acelerar a segmentação sem perder silhueta


@dataclass
class PhoneRecord:
    animal_id: str
    weight_kg: float | None
    file: Path
    view: str = "lateral"
    distance_m: float | None = None
    marker_px: float | None = None
    marker_m: float | None = None
    height_cm: float | None = None
    meta: dict | None = None


def extract_frames(video: str | Path, every_n: int = 6, max_frames: int = 40) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(f)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def _resize(img: np.ndarray) -> tuple[np.ndarray, float]:
    s = MAX_SIDE / max(img.shape[:2])
    if s >= 1:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def _focal_px(img_w: int, focal_35mm: float = 26.0) -> float:
    """Focal em pixels a partir da focal equivalente 35 mm (padrão da câmera principal
    dos celulares ≈ 24-28 mm). ARKit/Camera2 fornecem o valor exato; use quando houver."""
    return img_w * focal_35mm / 36.0


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_media(records: list[PhoneRecord], segmenter, every_n: int = 6,
                  max_frames: int = 40, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for rec in records:
        ext = rec.file.suffix.lower()
        if ext in VIDEO_EXT:
            frames = extract_frames(rec.file, every_n, max_frames)
        elif ext in IMAGE_EXT:
            img = cv2.imread(str(rec.file))
            frames = [img] if img is not None else []
        else:
            frames = []
        if not frames and verbose:
            print(f"{rec.file.name}: sem frames legíveis")
        for k, frame in enumerate(frames):
            img, s = _resize(frame)
            mask = segmenter.mask(img)
            if mask is None:
                continue
            h_px = int(mask.any(axis=1).sum())
            px_per_m, scale_src = None, "none"
            if rec.distance_m:
                px_per_m, scale_src = scale_from_distance(rec.distance_m, _focal_px(img.shape[1])), "distance"
            elif rec.marker_px and rec.marker_m:
                px_per_m, scale_src = scale_from_marker(rec.marker_px * s, rec.marker_m), "marker"
            elif rec.height_cm:
                px_per_m, scale_src = scale_from_known_height(h_px, rec.height_cm / 100), "known_height"
            try:
                f = silhouette_features(mask, px_per_m)
            except ValueError:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            q_touch = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
            q_blur = _blur_score(gray)
            q_ok = (not q_touch) and f["sil_fill"] > 0.35 and q_blur > 30
            rows.append({"animal_id": rec.animal_id, "weight_kg": rec.weight_kg, "file": rec.file.name,
                         "frame_idx": k, "view": rec.view, "scale_src": scale_src,
                         "px_per_m": px_per_m, "mask_h_px": h_px, **f,
                         "q_touches_border": q_touch, "q_blur": q_blur, "q_ok": q_ok,
                         **(rec.meta or {})})
        if verbose:
            n = sum(1 for r in rows if r["animal_id"] == rec.animal_id and r["file"] == rec.file.name)
            print(f"{rec.file.name}: {n} frames segmentados")
    return pd.DataFrame(rows)


def records_from_csv(csv_path: str | Path, media_dir: str | Path) -> list[PhoneRecord]:
    df = pd.read_csv(csv_path)
    media_dir = Path(media_dir)
    recs = []
    meta_cols = [c for c in df.columns if c in ("farm", "breed", "sex", "category")]
    for _, r in df.iterrows():
        def g(col):
            v = r.get(col)
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        recs.append(PhoneRecord(
            animal_id=str(r["animal_id"]), weight_kg=g("weight_kg"),
            file=media_dir / str(r["file"]), view=str(g("view") or "lateral"),
            distance_m=g("distance_m"), marker_px=g("marker_px"), marker_m=g("marker_m"),
            height_cm=g("height_cm"), meta={c: r[c] for c in meta_cols}))
    return recs
