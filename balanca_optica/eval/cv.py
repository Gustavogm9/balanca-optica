"""Validação cruzada agrupada por animal (ou por fazenda): a regra de ouro do projeto.

Nunca dividir por imagem: o mesmo animal em treino e teste infla o R² e esconde
o erro real (caso UFV 2015: R² 0,99 no treino, 19% de erro na validação).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ..model.regressor import WeightRegressor


def grouped_cv_predict(df: pd.DataFrame, kind: str, group_col: str = "animal_id",
                       target: str = "weight_kg", n_splits: int | None = None,
                       numeric: list[str] | None = None,
                       categorical: list[str] | None = None) -> np.ndarray:
    """Predição out-of-fold. n_splits=None -> leave-one-group-out."""
    groups = df[group_col].values
    y = df[target].values.astype(float)
    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    oof = np.full(len(df), np.nan)
    for tr, te in splitter.split(df, y, groups):
        kw = {}
        if numeric is not None:
            kw["numeric"] = numeric
        if categorical is not None:
            kw["categorical"] = categorical
        m = WeightRegressor(kind=kind, **kw).fit(df.iloc[tr], y[tr])
        oof[te] = m.predict(df.iloc[te])
    return oof
