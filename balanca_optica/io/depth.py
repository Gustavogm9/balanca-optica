"""Leitura de mapas de profundidade e conversão para coordenadas métricas."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import CameraIntrinsics


def load_depth(path: str | Path, intr: CameraIntrinsics) -> np.ndarray:
    """Carrega PNG 16 bits e devolve profundidade em metros (0 = inválido)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) * intr.depth_scale
    depth[raw == 0] = 0.0
    return depth


def depth_to_points(depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Mapa (H,W) em metros -> nuvem (H,W,3) em metros no referencial da câmera."""
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.dstack([x, y, z])


def pixel_area_map(depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Área real (m²) coberta por cada pixel, dada a profundidade: z² / (fx·fy)."""
    return (depth_m**2) / (intr.fx * intr.fy)
