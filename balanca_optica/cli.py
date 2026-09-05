"""CLI: bo extract | bo validate | bo train | bo predict | bo calibrate"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_intrinsics
from .eval.cv import grouped_cv_predict
from .eval.metrics import regression_metrics
from .eval.report import write_report
from .geometry.features import dorsal_features
from .io.depth import depth_to_points, load_depth
from .model.calibration import HerdCalibrator
from .model.regressor import WeightRegressor
from .model.uncertainty import ConformalInterval
from .segmentation.depth_seg import segment_animal


def cmd_extract(a):
    if a.dataset == "cowdatabase2":
        from .datasets.cowdatabase2 import extract_features
        df = extract_features(a.root, view=a.view)
    else:
        raise SystemExit("dataset desconhecido")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"salvo {a.out} ({len(df)} linhas)")


def cmd_validate(a):
    df = pd.read_csv(a.features)
    if a.only_ok and "q_ok" in df:
        df = df[df.q_ok].reset_index(drop=True)
    y = df.weight_kg.values
    results = {}
    for kind in a.models:
        p = grouped_cv_predict(df, kind, group_col=a.group, n_splits=a.folds)
        results[kind] = (y, p)
        print(kind, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in regression_metrics(y, p).items()}, ensure_ascii=False))
    results["média (baseline)"] = (y, np.full(len(y), y.mean()))
    path = write_report(a.out, a.title, results, notes=a.notes)
    print(f"relatório: {path}")


def cmd_train(a):
    df = pd.read_csv(a.features)
    if a.only_ok and "q_ok" in df:
        df = df[df.q_ok].reset_index(drop=True)
    y = df.weight_kg.values
    m = WeightRegressor(kind=a.model).fit(df, y)
    oof = grouped_cv_predict(df, a.model, group_col=a.group, n_splits=a.folds)
    ci = ConformalInterval(alpha=a.alpha).fit(y, oof)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    Path(str(a.out) + ".ci.json").write_text(json.dumps({"alpha": ci.alpha, "q_rel": ci.q_rel}))
    print(f"modelo salvo em {a.out}; IC {int((1-a.alpha)*100)}% = ±{ci.q_rel*100:.1f}% (n={len(y)})")


def cmd_predict(a):
    intr = get_intrinsics(a.camera)
    m = WeightRegressor.load(a.model)
    ci_path = Path(str(a.model) + ".ci.json")
    ci = None
    if ci_path.exists():
        d = json.loads(ci_path.read_text())
        ci = ConformalInterval(alpha=d["alpha"], q_rel=d["q_rel"])
    cal = None
    if a.calibration:
        d = json.loads(Path(a.calibration).read_text())
        cal = HerdCalibrator(**d)
    preds, ok = [], []
    for dp in a.depth:
        depth = load_depth(dp, intr)
        pts = depth_to_points(depth, intr)
        try:
            seg = segment_animal(pts, depth > 0)
            f = dorsal_features(pts, depth, seg, intr)
            p = float(m.predict(pd.DataFrame([f]))[0])
            preds.append(p); ok.append(seg.quality["ok"])
        except Exception as e:  # noqa: BLE001
            print(f"{dp}: sem leitura ({e})")
    from .aggregate import aggregate_frames
    agg = aggregate_frames(np.array(preds), np.array(ok))
    w = agg["weight"]
    if cal is not None and not np.isnan(w):
        w = float(cal.apply(np.array([w]))[0])
    out = {"weight_kg": None if np.isnan(w) else round(w, 1), **{k: v for k, v in agg.items() if k != "weight"}}
    if ci is not None and not np.isnan(w):
        lo, hi = ci.interval(np.array([w]), agg["n_frames"])
        out["ci_low_kg"], out["ci_high_kg"] = round(float(lo[0]), 1), round(float(hi[0]), 1)
    print(json.dumps(out, ensure_ascii=False))


def cmd_calibrate(a):
    df = pd.read_csv(a.pairs)  # colunas: weight_kg (balança), pred_kg (modelo)
    cal = HerdCalibrator(herd_id=a.herd).fit(df.weight_kg.values, df.pred_kg.values)
    Path(a.out).write_text(json.dumps(cal.to_dict(), indent=2))
    before = regression_metrics(df.weight_kg, df.pred_kg)["mape_pct"]
    after = regression_metrics(df.weight_kg, cal.apply(df.pred_kg.values))["mape_pct"]
    print(f"rebanho {a.herd}: a={cal.a:+.4f} b={cal.b:.4f} n={cal.n_pairs}; MAPE {before:.2f}% -> {after:.2f}% (in-sample)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bo", description="Balança Óptica - motor de estimativa de peso bovino")
    sp = ap.add_subparsers(dest="cmd", required=True)

    e = sp.add_parser("extract", help="extrai features de um dataset")
    e.add_argument("--dataset", default="cowdatabase2"); e.add_argument("--root", required=True)
    e.add_argument("--view", default="top"); e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_extract)

    v = sp.add_parser("validate", help="validação cruzada agrupada + relatório")
    v.add_argument("--features", required=True); v.add_argument("--out", required=True)
    v.add_argument("--models", nargs="+", default=["linear", "pls", "gbm"])
    v.add_argument("--group", default="animal_id"); v.add_argument("--folds", type=int, default=None)
    v.add_argument("--only-ok", action="store_true"); v.add_argument("--title", default="Validação")
    v.add_argument("--notes", default="")
    v.set_defaults(fn=cmd_validate)

    t = sp.add_parser("train", help="treina modelo final + intervalo conformal")
    t.add_argument("--features", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model", default="gbm"); t.add_argument("--group", default="animal_id")
    t.add_argument("--folds", type=int, default=None); t.add_argument("--alpha", type=float, default=0.10)
    t.add_argument("--only-ok", action="store_true")
    t.set_defaults(fn=cmd_train)

    p = sp.add_parser("predict", help="estima o peso a partir de 1+ depth.png (mesmo animal)")
    p.add_argument("--model", required=True); p.add_argument("--camera", default="kinect_v2")
    p.add_argument("--calibration", default=None); p.add_argument("depth", nargs="+")
    p.set_defaults(fn=cmd_predict)

    c = sp.add_parser("calibrate", help="calibração por rebanho a partir de pares balança/modelo")
    c.add_argument("--pairs", required=True); c.add_argument("--herd", required=True); c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_calibrate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
