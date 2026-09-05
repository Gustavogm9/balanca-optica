"""Features 2D de silhueta (caminho do celular sem profundidade).

Sem profundidade, o pipeline precisa de uma escala (px por metro) vinda do LiDAR,
de um marcador de tamanho conhecido ou da altura da câmera. Aqui as features são
calculadas em pixels e convertidas com `px_per_m`. Serve para o app (Fase 2) e
para vistas laterais.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

SILHOUETTE_FEATURES = ["sil_area", "sil_length", "sil_width", "sil_hull_area",
                       "sil_solidity", "sil_eccentricity", "sil_perimeter"]


def silhouette_features(mask: np.ndarray, px_per_m: float) -> dict[str, float]:
    lab = label(mask.astype(np.uint8))
    props = regionprops(lab)
    if not props:
        raise ValueError("máscara vazia")
    r = max(props, key=lambda p: p.area)
    s = 1.0 / px_per_m
    return {
        "sil_area": float(r.area * s * s),
        "sil_length": float(r.axis_major_length * s),
        "sil_width": float(r.axis_minor_length * s),
        "sil_hull_area": float(r.area_convex * s * s),
        "sil_solidity": float(r.solidity),
        "sil_eccentricity": float(r.eccentricity),
        "sil_perimeter": float(r.perimeter * s),
    }


def scale_from_marker(marker_px: float, marker_m: float) -> float:
    """px por metro a partir de um marcador de tamanho conhecido na mesma distância do animal."""
    return marker_px / marker_m


def scale_from_distance(distance_m: float, focal_px: float) -> float:
    """px por metro a partir da distância medida (LiDAR/ARKit) e da focal em pixels."""
    return focal_px / distance_m
