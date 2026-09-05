"""Features 2D de silhueta (caminho do celular sem profundidade).

Sem profundidade, o pipeline precisa de uma escala (px por metro) vinda do LiDAR,
de um marcador de tamanho conhecido ou da altura da câmera. As features são
calculadas em pixels e convertidas com `px_per_m`; sem escala (px_per_m=None)
saem normalizadas pela diagonal da imagem (só servem para comparar imagens
tiradas à mesma distância). Serve para o app (Fase 2) e para vistas laterais.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

SILHOUETTE_FEATURES = ["sil_area", "sil_length", "sil_width", "sil_hull_area", "sil_solidity",
                       "sil_eccentricity", "sil_perimeter", "sil_bbox_h", "sil_bbox_w",
                       "sil_body_area", "sil_body_h", "sil_fill"]


def _largest_region(mask: np.ndarray):
    lab = label(mask.astype(np.uint8))
    props = regionprops(lab)
    if not props:
        raise ValueError("máscara vazia")
    r = max(props, key=lambda p: p.area)
    return r, lab == r.label


def silhouette_features(mask: np.ndarray, px_per_m: float | None = None) -> dict[str, float]:
    """Features geométricas da maior região da máscara.

    sil_body_*: 'tronco' aproximado, removendo as pernas: mantém apenas as linhas
    da máscara cuja largura ocupada é ≥ 45% da largura máxima (as pernas são finas).
    """
    r, m = _largest_region(mask)
    if px_per_m is None:
        s = 1.0 / float(np.hypot(*mask.shape))  # normaliza pela diagonal da imagem
    else:
        s = 1.0 / px_per_m
    minr, minc, maxr, maxc = r.bbox
    sub = m[minr:maxr, minc:maxc]
    row_w = sub.sum(axis=1)
    body_rows = row_w >= 0.45 * row_w.max()
    body_area_px = int(sub[body_rows].sum())
    body_h_px = int(body_rows.sum())
    return {
        "sil_area": float(r.area * s * s),
        "sil_length": float(r.axis_major_length * s),
        "sil_width": float(r.axis_minor_length * s),
        "sil_hull_area": float(r.area_convex * s * s),
        "sil_solidity": float(r.solidity),
        "sil_eccentricity": float(r.eccentricity),
        "sil_perimeter": float(r.perimeter * s),
        "sil_bbox_h": float((maxr - minr) * s),
        "sil_bbox_w": float((maxc - minc) * s),
        "sil_body_area": float(body_area_px * s * s),
        "sil_body_h": float(body_h_px * s),
        "sil_fill": float(r.area / max((maxr - minr) * (maxc - minc), 1)),
    }


def scale_from_marker(marker_px: float, marker_m: float) -> float:
    """px por metro a partir de um marcador de tamanho conhecido na mesma distância do animal."""
    return marker_px / marker_m


def scale_from_distance(distance_m: float, focal_px: float) -> float:
    """px por metro a partir da distância medida (LiDAR/ARKit) e da focal em pixels."""
    return focal_px / distance_m


def scale_from_known_height(mask_height_px: float, animal_height_m: float) -> float:
    """px por metro quando a altura real do animal é conhecida (ex.: cernelha medida).
    Útil só para experimentos: simula o que o LiDAR entregaria."""
    return mask_height_px / animal_height_m
