"""Métricas do portão de saída: MAPE, MAE, RMSE, viés e limites de concordância."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    rel = err / y
    ss_res = float((err**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    return {
        "n": int(len(y)),
        "mape_pct": float(np.mean(np.abs(rel)) * 100),
        "mae_kg": float(np.mean(np.abs(err))),
        "rmse_kg": float(np.sqrt(np.mean(err**2))),
        "bias_kg": float(err.mean()),
        "bias_pct": float(rel.mean() * 100),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "p_within_3pct": float(np.mean(np.abs(rel) <= 0.03) * 100),
        "p_within_5pct": float(np.mean(np.abs(rel) <= 0.05) * 100),
        "y_range_kg": (float(y.min()), float(y.max())),
    }


def bland_altman(y_true, y_pred) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    diff = p - y
    mean = (p + y) / 2
    sd = diff.std(ddof=1)
    return {"bias_kg": float(diff.mean()), "sd_kg": float(sd),
            "loa_low_kg": float(diff.mean() - 1.96 * sd), "loa_high_kg": float(diff.mean() + 1.96 * sd),
            "mean_range_kg": (float(mean.min()), float(mean.max()))}


def lot_error(y_true, y_pred, lot_size: int = 30, n_rep: int = 500, seed: int = 0) -> dict[str, float]:
    """Erro do peso médio de lotes aleatórios de tamanho lot_size (bootstrap)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    k = min(lot_size, len(y))
    errs = []
    for _ in range(n_rep):
        idx = rng.choice(len(y), k, replace=False)
        errs.append((p[idx].mean() - y[idx].mean()) / y[idx].mean())
    errs = np.abs(np.array(errs)) * 100
    return {"lot_size": k, "lot_mape_pct": float(errs.mean()), "lot_p95_pct": float(np.percentile(errs, 95))}
