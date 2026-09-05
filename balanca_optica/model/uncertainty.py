"""Intervalo de confiança por leitura via conformal prediction (split conformal).

Usa resíduos relativos |y - ŷ|/ŷ de um conjunto de calibração (ou dos folds de
validação cruzada) e devolve o quantil (1-α) como semi-largura relativa. É livre
de suposições sobre a distribuição e tem cobertura garantida em média.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalInterval:
    alpha: float = 0.10
    q_rel: float | None = None  # semi-largura relativa calibrada

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalInterval":
        r = np.abs(np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_pred)
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n)
        self.q_rel = float(np.sort(r)[k - 1])
        return self

    def interval(self, y_pred: np.ndarray, n_frames: int | np.ndarray = 1) -> tuple[np.ndarray, np.ndarray]:
        """Limites inferior/superior. Com agregação de n frames independentes o ruído de
        postura cai ~1/sqrt(n), mas o erro do modelo não; usamos um encolhimento
        conservador: q / (1 + 0.5·(sqrt(n) - 1))."""
        assert self.q_rel is not None, "intervalo não calibrado"
        n = np.asarray(n_frames, dtype=float)
        q = self.q_rel / (1 + 0.5 * (np.sqrt(n) - 1))
        y = np.asarray(y_pred)
        return y * (1 - q), y * (1 + q)

    def coverage(self, y_true, y_pred, n_frames=1) -> float:
        lo, hi = self.interval(y_pred, n_frames)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))
