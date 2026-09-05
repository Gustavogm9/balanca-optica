"""Calibração por rebanho: corrige viés local com poucas pesagens de balança.

Ajusta log(y) = a + b·log(ŷ) com encolhimento bayesiano para o modelo global
(a→0, b→1) quando há poucos pares. Com 20 a 30 animais o ajuste converge; com 3
ou 4 fica quase todo no prior. É o mecanismo que leva o erro de ~5% para ~2-3%
em fazenda nova, e vira feature de produto ("conecte sua balança").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HerdCalibrator:
    herd_id: str = "default"
    a: float = 0.0
    b: float = 1.0
    n_pairs: int = 0
    prior_bias: float = 3.0  # pares "virtuais" segurando o viés em 0
    prior_slope: float = 15.0  # pares "virtuais" segurando a escala em 1
    fit_scale: bool = True

    def fit(self, y_scale: np.ndarray, y_pred: np.ndarray) -> "HerdCalibrator":
        ly = np.log(np.asarray(y_scale, float))
        lp = np.log(np.asarray(y_pred, float))
        n = len(ly)
        self.n_pairs = int(n)
        if n == 0:
            return self
        if self.fit_scale and n >= 10:
            # regressão com ridge em torno do prior (a=0,b=1)
            # modelo centrado: (ly - lp) = alpha + (b-1)·(lp - lp_bar); prior alpha=0, b=1
            lp_bar = lp.mean()
            lp_c = lp - lp_bar
            r = ly - lp
            slope = (lp_c @ (r - r.mean())) / max(lp_c @ lp_c, 1e-9)
            self.b = float(1.0 + n / (n + self.prior_slope) * slope)
            alpha = n / (n + self.prior_bias) * float(r.mean())
            self.a = float(alpha - (self.b - 1.0) * lp_bar)
        else:
            bias = float(np.mean(ly - lp))
            self.a, self.b = n / (n + self.prior_bias) * bias, 1.0
        return self

    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        return np.exp(self.a + self.b * np.log(np.asarray(y_pred, float)))

    def to_dict(self) -> dict:
        return {"herd_id": self.herd_id, "a": self.a, "b": self.b, "n_pairs": self.n_pairs}
