"""Loader do CowDB (Ruchay et al. 2020), 154 Hereford, Kinect v2, vistas top/left/right.

Repositório: https://github.com/ruchaya/CowDB
Estrutura: <root>/<N>/raw/top/depth-<ts>.png, rgb-<ts>.png ; Manual_measurements.xlsx (aba 'result').
Faixa de peso 243 a 605 kg: a base pública mais útil para validar a vista dorsal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PRESETS
from ..geometry.features import dorsal_features
from ..io.depth import depth_to_points, load_depth
from ..segmentation.depth_seg import segment_animal


def load_labels(root: str | Path) -> pd.DataFrame:
    df = pd.read_excel(Path(root) / "Manual_measurements.xlsx", sheet_name="result")
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame({
        "frame": df["N"].astype(int),
        "weight_kg": df["live weithg"].astype(float),
        "withers_cm": df["withers height"], "hip_cm": df["hip height"],
        "chest_width_cm": df["chest width"], "heart_girth_cm": df["heart girth"],
        "body_length_cm": df["oblique body length"],
    })
    out = out.dropna(subset=["weight_kg"])
    out["animal_id"] = "cowdb_" + out["frame"].astype(str)
    out["farm"] = "nizhny_novgorod"
    out["breed"] = "hereford"
    out["sex"] = "F"
    return out.reset_index(drop=True)


def extract_features(root: str | Path, view: str = "top", verbose: bool = True) -> pd.DataFrame:
    root = Path(root)
    intr = PRESETS["kinect_v2"]
    labels = load_labels(root)
    rows = []
    for _, r in labels.iterrows():
        d = root / str(r.frame) / "raw" / view
        files = sorted(d.glob("depth-*.png")) if d.exists() else []
        if not files:
            continue
        try:
            depth = load_depth(files[0], intr)
            pts = depth_to_points(depth, intr)
            seg = segment_animal(pts, depth > 0)
            feats = dorsal_features(pts, depth, seg, intr)
            q = {f"q_{k}": v for k, v in seg.quality.items()}
            rows.append({**r.to_dict(), **feats, **q, "n_frames": 1})
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"animal {r.frame}: falhou ({e})")
    df = pd.DataFrame(rows)
    if verbose:
        print(f"{len(df)}/{len(labels)} animais extraídos; qualidade ok em {int(df['q_ok'].sum())}")
    return df
