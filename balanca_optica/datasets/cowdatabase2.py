"""Loader do CowDatabase2 (Ruchay et al.), 119 Angus, Kinect v2, vistas top/left/right.

Repositório: https://github.com/ruchaya/CowDatabase2
Estrutura: <root>/<frame>/raw_data/{top,left,right}/{depth.png,rgb.png,cloud.ply}
Planilha Database2022v2.xlsx: 'Frame number' -> pasta; 'Live weigth' (kg) e medidas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PRESETS
from ..geometry.features import dorsal_features
from ..io.depth import depth_to_points, load_depth
from ..segmentation.depth_seg import segment_animal


def load_labels(root: str | Path) -> pd.DataFrame:
    df = pd.read_excel(Path(root) / "Database2022v2.xlsx")
    df.columns = [str(c).strip() for c in df.columns]
    out = df[["Frame number", "Live weigth", "Withers height", "Hip height",
              "Chest width", "Chest heart", "Birthday"]].copy()
    out.columns = ["frame", "weight_kg", "withers_cm", "hip_cm", "chest_width_cm",
                   "heart_girth_cm", "birthday"]
    out = out.dropna(subset=["frame", "weight_kg"])
    out["frame"] = out["frame"].astype(int)
    out["animal_id"] = "cdb2_" + out["frame"].astype(str)
    out["farm"] = "voronezh"
    out["breed"] = "angus"
    out["sex"] = "F"
    return out.reset_index(drop=True)


def extract_features(root: str | Path, view: str = "top", verbose: bool = True) -> pd.DataFrame:
    """Roda segmentação + geometria em todos os frames rotulados. Devolve um DataFrame
    com uma linha por frame (features, qualidade e rótulo)."""
    root = Path(root)
    intr = PRESETS["kinect_v2"]
    labels = load_labels(root)
    rows = []
    for _, r in labels.iterrows():
        dpath = root / str(r.frame) / "raw_data" / view / "depth.png"
        if not dpath.exists():
            continue
        try:
            depth = load_depth(dpath, intr)
            pts = depth_to_points(depth, intr)
            seg = segment_animal(pts, depth > 0)
            feats = dorsal_features(pts, depth, seg, intr)
            q = {f"q_{k}": v for k, v in seg.quality.items()}
            rows.append({**r.to_dict(), **feats, **q, "n_frames": 1})
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"frame {r.frame}: falhou ({e})")
    df = pd.DataFrame(rows)
    if verbose:
        print(f"{len(df)}/{len(labels)} frames extraídos; qualidade ok em {int(df['q_ok'].sum())}")
    return df
