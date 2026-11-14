"""Regressores de peso sobre features geométricas.

Três famílias, todas com a mesma interface:
- 'linear': Ridge em log(features) -> log(peso). Modelo alométrico (peso ∝ volume^a·...).
- 'pls': PLS em log, como em Bresolin 2026 (venceu redes neurais com n≈200).
- 'gbm': gradient boosting (LightGBM se instalado, senão HistGradientBoosting do sklearn).
Metadados categóricos (raça, sexo, categoria) entram one-hot quando fornecidos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODEL_KINDS = ("linear", "pls", "gbm")

# features positivas onde o log faz sentido (escala alométrica)
DEFAULT_NUMERIC = [
    "area_body", "volume_body", "length_body", "width_p50", "width_p90",
    "height_mean", "height_p90", "hull_area_body", "eq_cyl_radius", "vol_over_area",
]


def _safe_log(x):
    return np.log(np.clip(np.asarray(x, dtype=float), 1e-6, None))


@dataclass
class WeightRegressor:
    kind: str = "linear"
    numeric: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC))
    categorical: list[str] = field(default_factory=list)
    log_target: bool = True
    pipeline: Pipeline | None = None
    version: str | None = None  # ex.: 'phone_v0_bai_2026-11'; gravado em versao_modelo

    def _build(self) -> Pipeline:
        num = Pipeline([("log", FunctionTransformer(_safe_log)), ("sc", StandardScaler())])
        steps = [("num", num, self.numeric)]
        if self.categorical:
            steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical))
        pre = ColumnTransformer(steps)
        if self.kind == "linear":
            est = RidgeCV(alphas=np.logspace(-3, 2, 20))
        elif self.kind == "pls":
            est = PLSRegression(n_components=min(4, len(self.numeric)))
        elif self.kind == "gbm":
            try:
                from lightgbm import LGBMRegressor  # type: ignore
                est = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7,
                                    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
                                    reg_lambda=1.0, verbose=-1)
            except ImportError:
                est = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03,
                                                    max_leaf_nodes=7, min_samples_leaf=8,
                                                    l2_regularization=1.0)
        else:
            raise ValueError(f"kind deve ser um de {MODEL_KINDS}")
        return Pipeline([("pre", pre), ("est", est)])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WeightRegressor":
        self.pipeline = self._build()
        yt = np.log(y) if self.log_target else np.asarray(y, float)
        self.pipeline.fit(X, yt)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None, "modelo não treinado"
        p = np.asarray(self.pipeline.predict(X)).ravel()
        return np.exp(p) if self.log_target else p

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "WeightRegressor":
        return joblib.load(path)
