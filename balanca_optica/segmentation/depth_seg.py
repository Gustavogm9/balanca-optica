"""Segmentação do animal a partir da profundidade (vista dorsal, câmera fixa).

Não depende de rede neural: estima o plano do piso por RANSAC, separa o que está
acima do piso, mantém o maior componente e remove cabeça/cauda pelo perfil de
largura ao longo do eixo do corpo. Serve de baseline robusto e de fallback para
o segmentador YOLO/SAM (segmentation.yolo).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class FloorPlane:
    normal: np.ndarray  # unitário, aponta para a câmera (altura positiva = acima do piso)
    d: float  # plano: n·p + d = 0
    inlier_ratio: float

    def height(self, points: np.ndarray) -> np.ndarray:
        """Altura (m) de cada ponto acima do piso. points: (..., 3)."""
        return points @ self.normal + self.d


def fit_floor_plane(points: np.ndarray, valid: np.ndarray, n_iter: int = 300,
                    tol: float = 0.03, sample: int = 20000, seed: int = 0) -> FloorPlane:
    """RANSAC de plano sobre os pontos válidos. O piso é a maior superfície plana."""
    rng = np.random.default_rng(seed)
    pts = points[valid]
    if len(pts) < 100:
        raise ValueError("Poucos pontos válidos para estimar o piso")
    if len(pts) > sample:
        pts = pts[rng.choice(len(pts), sample, replace=False)]
    best_n, best_d, best_in = None, 0.0, -1
    for _ in range(n_iter):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p0
        dist = np.abs(pts @ n + d)
        inl = int((dist < tol).sum())
        if inl > best_in:
            best_in, best_n, best_d = inl, n, d
    # refina por mínimos quadrados nos inliers
    dist = np.abs(pts @ best_n + best_d)
    inl = pts[dist < tol]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1]
    # orienta a normal para a câmera (origem): pontos do piso têm z grande; queremos altura>0 acima do piso
    if n @ c > 0:
        n = -n
    d = -n @ c
    return FloorPlane(normal=n, d=float(d), inlier_ratio=len(inl) / len(pts))


@dataclass
class SegmentationResult:
    mask_full: np.ndarray  # animal inteiro (bool)
    mask_body: np.ndarray  # tronco sem cabeça/cauda (bool)
    height_map: np.ndarray  # altura acima do piso (m), 0 fora do válido
    floor: FloorPlane
    axis: np.ndarray  # vetor unitário 2D (u,v) do eixo do corpo em pixels
    center: np.ndarray  # centro (u,v) do tronco
    width_profile: np.ndarray  # largura (m) por fatia ao longo do eixo (tronco)
    quality: dict


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def trim_head_tail(mask: np.ndarray, points: np.ndarray, floor: FloorPlane,
                   n_bins: int = 40, rel_width: float = 0.6):
    """Remove cabeça e cauda: fatias ao longo do eixo principal cuja largura métrica
    cai abaixo de rel_width × largura mediana do terço central são descartadas a
    partir das extremidades."""
    vs, us = np.nonzero(mask)
    if len(us) < 50:
        return mask.copy(), np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.zeros(1)
    # projeta no plano do piso para medir em metros
    p = points[vs, us]
    h = floor.height(p)
    p_floor = p - np.outer(h, floor.normal)  # projeção ortogonal no piso
    c = p_floor.mean(axis=0)
    q = p_floor - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    t = q @ e1
    s = q @ e2
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    bins = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    width = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bins == b
        if sel.sum() >= 3:
            width[b] = np.percentile(s[sel], 97) - np.percentile(s[sel], 3)
    mid = width[n_bins // 3: 2 * n_bins // 3]
    ref = np.median(mid[mid > 0]) if (mid > 0).any() else width.max()
    keep = width >= rel_width * ref
    # mantém o bloco contíguo de bins largos ao redor do centro
    lo = n_bins // 2
    while lo - 1 >= 0 and keep[lo - 1]:
        lo -= 1
    hi = n_bins // 2
    while hi + 1 < n_bins and keep[hi + 1]:
        hi += 1
    body_sel = (bins >= lo) & (bins <= hi)
    body = np.zeros_like(mask)
    body[vs[body_sel], us[body_sel]] = True
    # eixo em pixels
    uc, vc = us.mean(), vs.mean()
    cov = np.cov(np.vstack([us - uc, vs - vc]))
    w, v = np.linalg.eigh(cov)
    axis_px = v[:, np.argmax(w)]
    return body, axis_px, np.array([us[body_sel].mean(), vs[body_sel].mean()]), width[lo:hi + 1]


def segment_animal(points: np.ndarray, valid: np.ndarray, h_min: float = 0.35,
                   h_max: float = 2.2, min_area_px: int = 2000,
                   floor: FloorPlane | None = None) -> SegmentationResult:
    """Segmenta o animal em uma vista dorsal RGB-D usando só geometria."""
    if floor is None:
        floor = fit_floor_plane(points, valid)
    height = floor.height(points)
    height[~valid] = 0.0
    cand = valid & (height > h_min) & (height < h_max)
    cand = ndi.binary_opening(cand, iterations=2)
    cand = ndi.binary_closing(cand, iterations=3)
    full = _largest_component(cand)
    full = ndi.binary_fill_holes(full)
    if full.sum() < min_area_px:
        raise ValueError(f"Animal não encontrado (área {int(full.sum())} px < {min_area_px})")
    body, axis, center, wprof = trim_head_tail(full, points, floor)

    # qualidade do frame
    h_img, w_img = full.shape
    border = np.zeros_like(full)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches_border = bool((full & border).any())
    # buracos de profundidade dentro do animal
    filled = ndi.binary_fill_holes(cand) if cand.any() else cand
    hole_ratio = float(1 - (cand & full).sum() / max(full.sum(), 1))
    # retidão: quanto a largura varia ao longo do tronco (animal curvo -> perfil irregular)
    straightness = float(1 - np.std(wprof) / (np.mean(wprof) + 1e-6)) if len(wprof) > 2 else 0.0
    body_frac = float(body.sum() / max(full.sum(), 1))
    quality = {
        "touches_border": touches_border,
        "hole_ratio": hole_ratio,
        "straightness": straightness,
        "body_fraction": body_frac,
        "floor_inlier_ratio": floor.inlier_ratio,
        "ok": (not touches_border) and hole_ratio < 0.15 and straightness > 0.7 and 0.45 < body_frac < 0.95,
    }
    return SegmentationResult(full, body, height, floor, axis, center, wprof, quality)


def overlay(rgb_or_depth_vis: np.ndarray, seg: SegmentationResult) -> np.ndarray:
    """Desenha máscara completa (azul) e tronco (verde) para inspeção."""
    img = rgb_or_depth_vis.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[seg.mask_full] = (0.5 * img[seg.mask_full] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
    img[seg.mask_body] = (0.5 * img[seg.mask_body] + 0.5 * np.array([0, 200, 60])).astype(np.uint8)
    return img
