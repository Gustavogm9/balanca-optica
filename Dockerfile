FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/yolo
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch CPU antes do ultralytics para não puxar as rodas CUDA (imagem ~5x menor)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY balanca_optica ./balanca_optica
RUN pip install ".[gbm,yolo,worker,serve]"

# pesos do segmentador baixados no build (fica offline em produção)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

COPY models/phone_v0_bai.joblib models/phone_v0_bai.joblib.ci.json ./models/

EXPOSE 8000
CMD ["bo", "worker", "--loop"]
