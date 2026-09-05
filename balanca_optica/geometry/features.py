"""Features geométricas métricas de uma vista dorsal RGB-D segmentada.

Método: os pontos do animal são projetados no plano do piso e rasterizados em uma
grade métrica (células de `cell` m) guardando a altura máxima por célula. Área e
volume vêm da grade (Σ célula, Σ altura·célula), o que corrige o viés que a fórmula
por pixel (z²/fx·fy) tem nas bordas inclinadas do corpo. Comprimento, larguras e
alturas saem da mesma grade, alinhada ao eixo principal do tronco.

Unidades SI (m, m², m³). São as variáveis que carregam o peso na literatura
(Bresolin 2026; Cominotte 2023; Weber 2020).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from ..config import CameraIntrinsics
from ..segmentation.depth_seg import SegmentationResult

FEATURE_NAMES = [
    "area_full", "area_body", "volume_full", "volume_body",
    "length_body", "width_p50", "width_p90", "width_max",
    "height_mean", "height_p90", "height_max", "height_std",
    "hull_area_body", "solidity_body", "elongation_body",
    "vol_over_area", "eq_cyl_radius", "camera_height",
]


def _floor_frame(points_body: np.ndarray, floor):
    """Base ortonormal no piso: e1 ao longo do tronco, e2 transversal; origem no centróide."""
    h = floor.height(points_body)
    p_floor = points_body - np.outer(h, floor.normal)
    c = p_floor.mean(axis=0)
    _, _, vt = np.linalg.svd(p_floor - c, full_matrices=False)
    e1 = vt[0] - (vt[0] @ floor.normal) * floor.normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(floor.normal, e1)
    return c, e1, e2


def _rasterize(points: np.ndarray, floor, c, e1, e2, cell: float):
    """Grade de altura máxima por célula do piso. Devolve (grid_h, occupancy, t_edges, s_edges)."""
    h = floor.height(points)
    p_floor = points - np.outer(h, floor.normal)
    q = p_floor - c
    t, s = q @ e1, q @ e2
    ti = np.floor((t - t.min()) / cell).astype(int)
    si = np.floor((s - s.min()) / cell).astype(int)
    pad = 3
    grid = np.full((ti.max() + 1 + 2 * pad, si.max() + 1 + 2 * pad), -np.inf)
    np.maximum.at(grid, (ti + pad, si + pad), h)
    occ = np.isfinite(grid)
    # fecha lacunas de amostragem (bordas inclinadas têm poucos pontos por célula)
    occ_c = occ | ndi.binary_fill_holes(ndi.binary_closing(occ, iterations=2))
    idx = ndi.distance_transform_edt(~occ, return_distances=False, return_indices=True)
    grid = np.where(occ_c, grid[idx[0], idx[1]], 0.0)
    return grid, occ_c, t, s


def dorsal_features(points: np.ndarray, depth_m: np.ndarray, seg: SegmentationResult,
                    intr: CameraIntrinsics, cell: float = 0.01) -> dict[str, float]:
    full, body = seg.mask_full, seg.mask_body
    floor = seg.floor
    c, e1, e2 = _floor_frame(points[body], floor)
    ca = cell * cell

    g_full, o_full, _, _ = _rasterize(points[full], floor, c, e1, e2, cell)
    g_body, o_body, t, s = _rasterize(points[body], floor, c, e1, e2, cell)

    f: dict[str, float] = {}
    f["area_full"] = float(o_full.sum() * ca)
    f["area_body"] = float(o_body.sum() * ca)
    f["volume_full"] = float(g_full[o_full].sum() * ca)
    f["volume_body"] = float(g_body[o_body].sum() * ca)

    # comprimento e perfil de largura a partir da grade do tronco (linhas = eixo t)
    rows = o_body.any(axis=1)
    f["length_body"] = float(rows.sum() * cell)
    widths = o_body.sum(axis=1)[rows] * cell
    f["width_p50"] = float(np.percentile(widths, 50))
    f["width_p90"] = float(np.percentile(widths, 90))
    f["width_max"] = float(widths.max())

    hb = g_body[o_body]
    f["height_mean"] = float(hb.mean())
    f["height_p90"] = float(np.percentile(hb, 90))
    f["height_max"] = float(np.percentile(hb, 99.5))
    f["height_std"] = float(hb.std())

    try:
        hull = ConvexHull(np.c_[t, s])
        f["hull_area_body"] = float(hull.volume)  # em 2D, .volume é a área
    except Exception:
        f["hull_area_body"] = f["area_body"]
    f["solidity_body"] = float(f["area_body"] / max(f["hull_area_body"], 1e-6))
    f["elongation_body"] = float(f["length_body"] / max(f["width_p50"], 1e-6))
    f["vol_over_area"] = float(f["volume_body"] / max(f["area_body"], 1e-6))
    f["eq_cyl_radius"] = float(np.sqrt(f["volume_body"] / (np.pi * max(f["length_body"], 1e-6))))
    f["camera_height"] = float(abs(floor.d))
    return f
