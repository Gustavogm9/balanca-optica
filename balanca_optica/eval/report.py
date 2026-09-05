"""Relatório de validação em Markdown + figuras (dispersão e Bland-Altman)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import bland_altman, lot_error, regression_metrics  # noqa: E402


def write_report(out_dir: str | Path, title: str, results: dict[str, tuple[np.ndarray, np.ndarray]],
                 notes: str = "") -> Path:
    """results: {nome_do_modelo: (y_true, y_pred)}"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if notes:
        lines += [notes, ""]
    lines += ["| modelo | n | MAPE % | MAE kg | RMSE kg | viés % | R² | ≤3% | ≤5% | lote(30) MAPE % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (y, p) in results.items():
        m = regression_metrics(y, p)
        lo = lot_error(y, p)
        lines.append(f"| {name} | {m['n']} | {m['mape_pct']:.2f} | {m['mae_kg']:.1f} | {m['rmse_kg']:.1f} | "
                     f"{m['bias_pct']:+.2f} | {m['r2']:.2f} | {m['p_within_3pct']:.0f}% | {m['p_within_5pct']:.0f}% | "
                     f"{lo['lot_mape_pct']:.2f} |")
    lines.append("")
    # figuras
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in results.items():
        axes[0].scatter(y, p, s=18, alpha=.7, label=name)
    lim = [min(min(y.min(), p.min()) for y, p in results.values()) - 10,
           max(max(y.max(), p.max()) for y, p in results.values()) + 10]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("balança (kg)"); axes[0].set_ylabel("estimado (kg)"); axes[0].legend(); axes[0].set_title("Estimado vs balança")
    name, (y, p) = next(iter(results.items()))
    ba = bland_altman(y, p)
    axes[1].scatter((y + p) / 2, p - y, s=18, alpha=.7)
    for v, ls in [(ba["bias_kg"], "-"), (ba["loa_low_kg"], "--"), (ba["loa_high_kg"], "--")]:
        axes[1].axhline(v, color="k", ls=ls, lw=1)
    axes[1].set_xlabel("média (kg)"); axes[1].set_ylabel("estimado − balança (kg)")
    axes[1].set_title(f"Bland-Altman ({name}): viés {ba['bias_kg']:+.1f} kg, LoA ±{1.96*ba['sd_kg']:.1f} kg")
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=130)
    plt.close(fig)
    lines += ["![validação](validation.png)", ""]
    path = out / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
