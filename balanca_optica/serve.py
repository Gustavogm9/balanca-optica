"""API HTTP mínima (FastAPI) para testar sem Supabase ou para o Pasto Leve chamar direto.

POST /estimate  multipart: file (foto ou vídeo) + opcionais distance_m, marker_m, height_cm
                -> JSON de `estimate_from_media`
GET  /health    -> {"status": "ok", "model_version": ...}

`bo serve --port 8000`. Config: BO_MODEL, BO_SEG_WEIGHTS.
"""

import os
import shutil
import tempfile
from pathlib import Path

from .phone.estimate import estimate_from_media, load_model_bundle, reliability_score
from .phone.import_media import IMAGE_EXT, VIDEO_EXT


def create_app(model_path: str | None = None, seg_weights: str | None = None,
               model=None, segmenter=None, ci=None, model_version: str | None = None):
    """Monta o app. Nos testes passe `model`/`segmenter` prontos; em produção carrega do disco."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    model_path = model_path or os.environ.get("BO_MODEL", "models/phone_v0_bai.joblib")
    seg_weights = seg_weights or os.environ.get("BO_SEG_WEIGHTS", "yolov8n-seg.pt")
    state = {"model": model, "segmenter": segmenter, "ci": ci, "version": model_version}

    def _load():
        if state["model"] is None:
            state["model"], state["ci"], state["version"] = load_model_bundle(model_path)
        if state["segmenter"] is None:
            from .segmentation.yolo import YoloSegmenter
            state["segmenter"] = YoloSegmenter(seg_weights, conf=0.3)
        return state

    app = FastAPI(title="Balança Óptica", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": state["version"] or Path(model_path).stem}

    @app.post("/estimate")
    async def estimate(file: UploadFile = File(...), distance_m: float | None = Form(None),
                       marker_m: float | None = Form(None), height_cm: float | None = Form(None),
                       every_n: int = Form(6), max_frames: int = Form(40)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXT | IMAGE_EXT:
            raise HTTPException(415, f"extensão não suportada: {ext or '(vazia)'}")
        st = _load()
        with tempfile.TemporaryDirectory(prefix="bo_api_") as td:
            local = Path(td) / f"media{ext}"
            with local.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            res = estimate_from_media(local, st["model"], st["segmenter"], ci=st["ci"],
                                      every_n=every_n, max_frames=max_frames, distance_m=distance_m,
                                      marker_m=marker_m, height_cm=height_cm, model_version=st["version"])
        res["confiabilidade"] = reliability_score(res)
        return res

    return app


def app():  # alvo para `uvicorn balanca_optica.serve:app --factory`
    return create_app()
