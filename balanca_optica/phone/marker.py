"""Detecção automática do marcador de escala (ArUco) em fotos e vídeos de celular.

O marcador é uma folha A4 impressa com um ArUco (dicionário 4x4_50, id 0 a 9) de
lado conhecido (padrão 0,16 m, cabe em A4 com margem). Ele fica encostado no animal
ou na cerca, à mesma distância da câmera. A partir dos 4 cantos detectados estimamos
px por metro (média dos 4 lados), com correção de perspectiva pela razão entre lados.

Gere o marcador para impressão com `bo marker --out marcador.png` (ou via CLI).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
DEFAULT_SIDE_M = 0.16


@dataclass
class MarkerHit:
    marker_id: int
    corners: np.ndarray  # (4,2) px
    side_px: float
    px_per_m: float
    skew: float  # razão menor/maior lado; < 0,7 = muito inclinado, não confiar


def _detector():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMax = 53
    p.minMarkerPerimeterRate = 0.01
    return cv2.aruco.ArucoDetector(d, p)


def detect_marker(img_bgr: np.ndarray, side_m: float = DEFAULT_SIDE_M) -> MarkerHit | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    best = None
    for c, i in zip(corners, ids.ravel()):
        pts = c.reshape(4, 2)
        sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        side_px = float(sides.mean())
        skew = float(sides.min() / sides.max())
        hit = MarkerHit(int(i), pts, side_px, side_px / side_m, skew)
        if best is None or hit.side_px > best.side_px:
            best = hit
    return best


def generate_marker(marker_id: int = 0, side_px: int = 1200, border_px: int = 120) -> np.ndarray:
    """Imagem para impressão em A4: marcador + instrução. Lado impresso deve medir 0,16 m
    (a régua na folha serve para conferir após imprimir)."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    m = cv2.aruco.generateImageMarker(d, marker_id, side_px)
    h = side_px + 2 * border_px + 260
    w = side_px + 2 * border_px
    canvas = np.full((h, w), 255, np.uint8)
    canvas[border_px:border_px + side_px, border_px:border_px + side_px] = m
    # régua de 10 cm proporcional (1200 px = 16 cm -> 750 px = 10 cm)
    y = border_px + side_px + 90
    x0 = border_px
    x1 = x0 + int(side_px * 0.10 / DEFAULT_SIDE_M)
    cv2.line(canvas, (x0, y), (x1, y), 0, 6)
    for k in range(11):
        x = x0 + int((x1 - x0) * k / 10)
        cv2.line(canvas, (x, y - (18 if k % 5 == 0 else 10)), (x, y), 0, 4)
    cv2.putText(canvas, "regua 10 cm - confira apos imprimir", (x0, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.putText(canvas, f"Balanca Optica - marcador id {marker_id} - lado {int(DEFAULT_SIDE_M*100)} cm",
                (x0, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.putText(canvas, "Imprimir em A4 sem 'ajustar a pagina' (100%). Plastificar.", (x0, y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return canvas
