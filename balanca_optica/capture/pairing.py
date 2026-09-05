"""Pareamento clipe <-> pesagem da balança para montar o dataset de treino.

Entrada: readings.csv do daemon e um CSV da balança com colunas
`ts` (epoch s ou ISO 8601), `weight_kg`, opcional `tag` (RFID/brinco), `animal_id`.
Regras: mesmo tag quando houver; senão o clipe mais próximo no tempo dentro de
`window_s`. Saída: pairs.csv com clip_id, animal_id, weight_kg, dt_s, path.
"""
from __future__ import annotations

import pandas as pd


def pair_clips_with_scale(readings_csv: str, scale_csv: str, window_s: float = 90.0) -> pd.DataFrame:
    r = pd.read_csv(readings_csv)
    s = pd.read_csv(scale_csv)
    if not pd.api.types.is_numeric_dtype(s["ts"]):
        s["ts"] = pd.to_datetime(s["ts"]).astype("int64") / 1e9
    r["ts_mid"] = (r["ts_start"] + r["ts_end"]) / 2
    rows = []
    used = set()
    for _, w in s.sort_values("ts").iterrows():
        cand = r[~r["clip_id"].isin(used)].copy()
        if "tag" in s.columns and pd.notna(w.get("tag")) and "rfid" in cand.columns:
            same = cand[cand["rfid"].astype(str) == str(w["tag"])]
            if len(same):
                cand = same
        cand["dt"] = (cand["ts_mid"] - w["ts"]).abs()
        cand = cand[cand["dt"] <= window_s]
        if cand.empty:
            continue
        best = cand.sort_values("dt").iloc[0]
        used.add(best["clip_id"])
        rows.append({"clip_id": best["clip_id"], "animal_id": w.get("animal_id", w.get("tag")),
                     "weight_kg": w["weight_kg"], "dt_s": round(float(best["dt"]), 1), "path": best["path"],
                     "pred_kg": best.get("weight_kg")})
    return pd.DataFrame(rows)
