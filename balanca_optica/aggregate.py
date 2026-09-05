"""Agregação multi-frame: uma leitura por animal a partir de várias estimativas."""
from __future__ import annotations

import numpy as np


def aggregate_frames(preds: np.ndarray, quality_ok: np.ndarray | None = None,
                     min_frames: int = 3) -> dict:
    """Mediana robusta das estimativas dos frames aprovados no filtro de qualidade.
    Devolve peso, dispersão (MAD relativo) e contagem, para alimentar o intervalo."""
    p = np.asarray(preds, float)
    if quality_ok is not None:
        p = p[np.asarray(quality_ok, bool)]
    if len(p) == 0:
        return {"weight": np.nan, "n_frames": 0, "rel_mad": np.nan, "reliable": False}
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) * 1.4826
    return {"weight": med, "n_frames": int(len(p)), "rel_mad": mad / med,
            "reliable": len(p) >= min_frames and mad / med < 0.06}
