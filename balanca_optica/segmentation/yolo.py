"""Segmentador neural opcional (Ultralytics YOLO-seg) para RGB e para o celular.

Instale com `pip install balanca-optica[yolo]`. Sem GPU roda em CPU (~0,3 s/frame
com yolov8n-seg). O modelo COCO já tem a classe 'cow' (id 19); para Nelore em
curral o fine-tuning com ~300 máscaras anotadas é o primeiro passo da Fase 1.
"""
from __future__ import annotations

import numpy as np

COCO_COW = 19


class YoloSegmenter:
    def __init__(self, weights: str = "yolov8n-seg.pt", conf: float = 0.4, device: str | None = None):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("instale com: pip install 'balanca-optica[yolo]'") from e
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        names = self.model.names
        # modelo fine-tunado (1 classe 'bovino') -> classe 0; COCO -> 'cow' (19)
        self.class_id = 0 if len(names) == 1 else next((i for i, n in names.items() if str(n).lower() in ("cow", "bovino")), COCO_COW)

    def mask(self, rgb_bgr: np.ndarray, class_id: int | None = None) -> np.ndarray | None:
        """Devolve a máscara booleana do maior animal detectado, ou None."""
        class_id = self.class_id if class_id is None else class_id
        res = self.model.predict(rgb_bgr, conf=self.conf, device=self.device, verbose=False)[0]
        if res.masks is None:
            return None
        best, best_area = None, 0
        for m, c in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if int(c) != class_id:
                continue
            area = int(m.sum())
            if area > best_area:
                best, best_area = m, area
        if best is None:
            return None
        import cv2
        return cv2.resize(best.astype(np.uint8), (rgb_bgr.shape[1], rgb_bgr.shape[0])) > 0
