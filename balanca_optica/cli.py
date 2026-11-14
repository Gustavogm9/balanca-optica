"""CLI: bo extract | validate | train | predict | calibrate | capture | pair | import-phone | worker | serve"""
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
    elif a.dataset == "cowdb":
        from .datasets.cowdb import extract_features
        df = extract_features(a.root, view=a.view)
    elif a.dataset in ("bai", "rahman"):
        from .datasets.mendeley import bai_records, rahman_records, extract
        df = extract(bai_records(a.root) if a.dataset == "bai" else rahman_records(a.root))
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
    if a.view and "view" in df:
        df = df[df.view == a.view].reset_index(drop=True)
    if a.with_scale and "scale_src" in df:
        df = df[df.scale_src != "none"].reset_index(drop=True)
    y = df.weight_kg.values
    kw = {}
    if a.numeric:
        kw["numeric"] = list(a.numeric)
    if a.categorical:
        kw["categorical"] = list(a.categorical)
    m = WeightRegressor(kind=a.model, version=a.version, **kw).fit(df, y)
    oof = grouped_cv_predict(df, a.model, group_col=a.group, n_splits=a.folds, **kw)
    ci = ConformalInterval(alpha=a.alpha).fit(y, oof)
    met = regression_metrics(y, oof)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    Path(str(a.out) + ".ci.json").write_text(json.dumps({
        "alpha": ci.alpha, "q_rel": ci.q_rel, "model_version": a.version, "kind": a.model,
        "numeric": m.numeric, "n": int(len(y)), "n_groups": int(df[a.group].nunique()) if a.group in df else None,
        "mape_pct": round(met["mape_pct"], 2), "mae_kg": round(met["mae_kg"], 1), "r2": round(met["r2"], 3),
        "coverage": round(ci.coverage(y, oof), 3)}, indent=2))
    print(f"modelo salvo em {a.out}; MAPE out-of-fold {met['mape_pct']:.2f}% ; "
          f"IC {int((1-a.alpha)*100)}% = ±{ci.q_rel*100:.1f}% (n={len(y)})")


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


def cmd_capture(a):
    from .capture.sources import open_source
    from .capture.daemon import CaptureDaemon, CaptureConfig
    cfg = CaptureConfig(out_dir=a.out, site=a.site, model_path=a.model, h_min=a.h_min,
                        presence_on=a.on, presence_off=a.off, max_clip_s=a.max_clip)
    src = open_source(a.source)
    d = CaptureDaemon(src, cfg, on_reading=lambda m: print(json.dumps(m["estimate"] | {"clip": m["clip_id"], "n": m["n_frames"]})))
    print(f"capturando de {a.source} -> {a.out} (Ctrl+C para parar)")
    d.run(max_frames=a.max_frames)


def cmd_pair(a):
    from .capture.pairing import pair_clips_with_scale
    df = pair_clips_with_scale(a.readings, a.scale, a.window)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} pares gravados em {a.out}")


def cmd_import_phone(a):
    from .phone.import_media import records_from_csv, process_media
    from .segmentation.yolo import YoloSegmenter
    recs = records_from_csv(a.csv, a.media)
    df = process_media(recs, YoloSegmenter(a.weights, conf=0.3), every_n=a.every, max_frames=a.max_frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} frames de {df.animal_id.nunique() if len(df) else 0} animais -> {a.out}")


def cmd_marker(a):
    import cv2
    from .phone.marker import generate_marker
    from PIL import Image
    from .phone.marker import DEFAULT_SIDE_M
    img = generate_marker(a.id, side_px=1200)
    dpi = 1200 / (DEFAULT_SIDE_M * 100 / 2.54)  # px do lado / polegadas do lado
    pil = Image.fromarray(img)
    out = Path(a.out)
    pil.save(out, dpi=(dpi, dpi))
    pdf = out.with_suffix(".pdf")
    # página A4 (210x297 mm) com o marcador centralizado, escala física exata
    a4 = Image.new("1", (int(8.27 * dpi), int(11.69 * dpi)), 255)
    a4.paste(pil.convert("1"), ((a4.width - pil.width) // 2, (a4.height - pil.height) // 2))
    a4.save(pdf, resolution=dpi)
    print(f"marcador id {a.id}: {out} e {pdf} (A4, imprimir a 100%; lado do quadrado = {DEFAULT_SIDE_M*100:.0f} cm)")


def cmd_seg(a):
    from .segmentation import finetune as ft
    if a.action == "pseudo":
        print("data.yaml:", ft.pseudo_label(a.images, a.out, weights=a.weights))
    elif a.action == "merge":
        print("data.yaml:", ft.merge(a.sources, a.out))
    elif a.action == "train":
        best = ft.train(a.data, weights=a.weights, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device)
        print("best.pt:", best); print(json.dumps(ft.evaluate(best, a.data, imgsz=a.imgsz, device=a.device)))
    elif a.action == "eval":
        print(json.dumps(ft.evaluate(a.weights, a.data, imgsz=a.imgsz, device=a.device)))


def cmd_auction(a):
    from .datasets import auction as au
    if a.action == "discover":
        d = au.discover(a.url, a.out)
        if d["pdf"]:
            lots = au.parse_catalog(d["pdf"], hint=a.url.rsplit("/", 1)[-1].replace("-", " "))
            for l in lots:  # completa vídeo a partir da página quando o PDF não tem o link
                if not l.video_id:
                    l.video_id = d["videos"].get(l.lot.lstrip("0") or "0")
            print("csv:", au.write_lots_csv(lots, Path(a.out) / "lotes.csv", source=a.url) if lots else "nenhum lote com peso (leilão de genética/prenhez?)")
    elif a.action == "parse":
        lots = au.parse_catalog(a.pdf, hint=a.hint)
        print("csv:", au.write_lots_csv(lots, a.out, source=a.source or Path(a.pdf).name))
    elif a.action == "fetch":
        got = au.fetch_videos(a.csv, a.out, cookies_from_browser=a.cookies, limit=a.limit)
        print(f"{len(got)} vídeos disponíveis em {a.out}")
    elif a.action == "build":
        au.build_collection_csv(a.csv, a.videos, a.out, farm=a.farm, ref_date=a.ref_date, include_lots=a.include_lots)


def cmd_worker(a):
    from .worker.supabase_worker import main as worker_main
    worker_main(once=a.once, limit=a.limit, poll_seconds=a.poll)


def cmd_serve(a):
    import uvicorn
    from .serve import create_app
    uvicorn.run(create_app(model_path=a.model, seg_weights=a.weights), host=a.host, port=a.port)


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
    t.add_argument("--numeric", nargs="+", default=None, help="features numéricas (ex.: silhueta: sil_area sil_length ...)")
    t.add_argument("--categorical", nargs="*", default=None)
    t.add_argument("--view", default=None, help="filtra a coluna view (ex.: lateral)")
    t.add_argument("--with-scale", action="store_true", help="descarta frames com scale_src=none")
    t.add_argument("--version", default=None, help="nome da versão gravado no modelo (versao_modelo)")
    t.set_defaults(fn=cmd_train)

    p = sp.add_parser("predict", help="estima o peso a partir de 1+ depth.png (mesmo animal)")
    p.add_argument("--model", required=True); p.add_argument("--camera", default="kinect_v2")
    p.add_argument("--calibration", default=None); p.add_argument("depth", nargs="+")
    p.set_defaults(fn=cmd_predict)

    c = sp.add_parser("calibrate", help="calibração por rebanho a partir de pares balança/modelo")
    c.add_argument("--pairs", required=True); c.add_argument("--herd", required=True); c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_calibrate)

    k = sp.add_parser("capture", help="daemon do ponto fixo (kinect2|openni2|orbbec|realsense|replay:<pasta>)")
    k.add_argument("--source", required=True); k.add_argument("--out", default="capture_out")
    k.add_argument("--site", default="ponto_fixo"); k.add_argument("--model", default=None)
    k.add_argument("--h-min", type=float, default=0.35); k.add_argument("--on", type=float, default=0.04)
    k.add_argument("--off", type=float, default=0.01); k.add_argument("--max-clip", type=float, default=20.0)
    k.add_argument("--max-frames", type=int, default=None)
    k.set_defaults(fn=cmd_capture)

    pr = sp.add_parser("pair", help="pareia clipes (readings.csv) com pesagens da balança")
    pr.add_argument("--readings", required=True); pr.add_argument("--scale", required=True)
    pr.add_argument("--out", required=True); pr.add_argument("--window", type=float, default=90.0)
    pr.set_defaults(fn=cmd_pair)

    ip = sp.add_parser("import-phone", help="fotos/vídeos de celular + CSV de coleta -> features")
    ip.add_argument("--csv", required=True); ip.add_argument("--media", required=True); ip.add_argument("--out", required=True)
    ip.add_argument("--weights", default="yolov8n-seg.pt"); ip.add_argument("--every", type=int, default=6)
    ip.add_argument("--max-frames", type=int, default=40)
    ip.set_defaults(fn=cmd_import_phone)

    mk = sp.add_parser("marker", help="gera o marcador ArUco de escala para imprimir em A4")
    mk.add_argument("--id", type=int, default=0); mk.add_argument("--out", default="marcador.png")
    mk.set_defaults(fn=cmd_marker)

    sg = sp.add_parser("seg", help="segmentador: pseudo | merge | train | eval")
    sg.add_argument("action", choices=["pseudo", "merge", "train", "eval"])
    sg.add_argument("--images", default=None); sg.add_argument("--sources", nargs="*", default=[])
    sg.add_argument("--out", default="data/seg"); sg.add_argument("--data", default=None)
    sg.add_argument("--weights", default="yolov8n-seg.pt"); sg.add_argument("--epochs", type=int, default=50)
    sg.add_argument("--imgsz", type=int, default=640); sg.add_argument("--batch", type=int, default=8)
    sg.add_argument("--device", default=None)
    sg.set_defaults(fn=cmd_seg)

    au = sp.add_parser("auction", help="catálogo de leilão (PDF) + vídeos por lote -> dataset: parse | fetch | build")
    au.add_argument("action", choices=["discover", "parse", "fetch", "build"])
    au.add_argument("--url", default=None, help="discover: página do leilão (programaleiloes.com/agenda/leilao/...)")
    au.add_argument("--pdf", default=None); au.add_argument("--csv", default=None); au.add_argument("--videos", default=None)
    au.add_argument("--out", required=True); au.add_argument("--source", default=None)
    au.add_argument("--cookies", default=None, help="ex.: chrome (passa --cookies-from-browser ao yt-dlp)")
    au.add_argument("--limit", type=int, default=None); au.add_argument("--farm", default="leilao")
    au.add_argument("--hint", default=None, help="parse: título/sexo do leilão (ex.: 'touros nelore') quando o PDF não diz")
    au.add_argument("--include-lots", action="store_true", help="build: inclui lotes com peso médio (modo lote)")
    au.add_argument("--ref-date", default=None, help="data da pesagem/leilão AAAA-MM-DD para calcular categoria")
    au.set_defaults(fn=cmd_auction)

    wk = sp.add_parser("worker", help="processa vídeos pendentes do Pasto Leve (Supabase) e grava a estimativa")
    g = wk.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="processa a fila uma vez e sai")
    g.add_argument("--loop", action="store_true", help="fica em loop (padrão)")
    wk.add_argument("--limit", type=int, default=10, help="registros por ciclo")
    wk.add_argument("--poll", type=float, default=None, help="segundos entre ciclos (BO_POLL_SECONDS)")
    wk.set_defaults(fn=cmd_worker)

    sv = sp.add_parser("serve", help="API HTTP: POST /estimate (mídia) e GET /health")
    sv.add_argument("--host", default="0.0.0.0"); sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--model", default=None, help="BO_MODEL (padrão models/phone_v0_bai.joblib)")
    sv.add_argument("--weights", default=None, help="BO_SEG_WEIGHTS (padrão yolov8n-seg.pt)")
    sv.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
