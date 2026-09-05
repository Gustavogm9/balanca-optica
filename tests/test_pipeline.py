import numpy as np
import pandas as pd
import pytest

from balanca_optica.config import CameraIntrinsics
from balanca_optica.io.depth import depth_to_points
from balanca_optica.segmentation.depth_seg import segment_animal
from balanca_optica.geometry.features import dorsal_features
from balanca_optica.geometry.silhouette import silhouette_features
from balanca_optica.model.regressor import WeightRegressor
from balanca_optica.model.uncertainty import ConformalInterval
from balanca_optica.model.calibration import HerdCalibrator
from balanca_optica.eval.metrics import regression_metrics, bland_altman
from balanca_optica.aggregate import aggregate_frames

INTR = CameraIntrinsics(fx=400, fy=400, cx=256, cy=212, width=512, height=424, depth_scale=0.001)


def synthetic_depth(length=1.6, width=0.6, height=0.7, belly=0.5, cam_h=3.2, head=True, noise=0.0, seed=0):
    """Tronco = semi-elipsoide (a,b,c) sobre um 'ventre' plano a `belly` m do piso
    (as pernas não aparecem na vista dorsal). Volume acima do piso, analítico:
    π·a·b·belly + (2/3)·π·a·b·c."""
    rng = np.random.default_rng(seed)
    h, w = INTR.height, INTR.width
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    a, b, c = length / 2, width / 2, height
    depth = np.full((h, w), cam_h, dtype=np.float32)
    # ray marching: primeiro z ao longo do raio de cada pixel em que o ponto está dentro do corpo
    found = np.zeros((h, w), bool)
    for z in np.linspace(cam_h - belly - c, cam_h, 300):
        x = (us - INTR.cx) * z / INTR.fx
        y = (vs - INTR.cy) * z / INTR.fy
        r2 = (x / a) ** 2 + (y / b) ** 2
        surf = np.where(r2 < 1, belly + c * np.sqrt(np.clip(1 - r2, 0, 1)), 0.0)
        if head:  # cabeça: bloco estreito e mais baixo saindo da frente
            surf = np.where((x > a) & (x < a + 0.45) & (np.abs(y) < 0.12), 0.45, surf)
        hit = (~found) & (cam_h - z <= surf) & (surf > 0)
        depth[hit] = z
        found |= hit
    if noise:
        depth += rng.normal(0, noise, depth.shape).astype(np.float32)
    vol = np.pi * a * b * belly + (2 / 3) * np.pi * a * b * c
    if head:
        vol += 0.45 * 0.24 * 0.45
    return depth, vol


def test_segmentation_and_volume_recovery():
    depth, vol_true = synthetic_depth()
    pts = depth_to_points(depth, INTR)
    seg = segment_animal(pts, depth > 0)
    f = dorsal_features(pts, depth, seg, INTR)
    assert seg.quality["ok"]
    assert f["volume_full"] == pytest.approx(vol_true, rel=0.08)
    # o tronco é aparado onde a largura cai abaixo de 60% da largura central
    assert 1.15 <= f["length_body"] <= 1.6 and f["volume_body"] < f["volume_full"]
    assert f["width_max"] == pytest.approx(0.6, rel=0.10)
    assert f["height_max"] == pytest.approx(1.2, rel=0.05)
    assert f["camera_height"] == pytest.approx(3.2, abs=0.03)
    # a cabeça foi removida do tronco
    assert seg.mask_body.sum() < seg.mask_full.sum()


def test_noise_robustness():
    depth, vol_true = synthetic_depth(noise=0.01, seed=1)
    pts = depth_to_points(depth, INTR)
    seg = segment_animal(pts, depth > 0)
    f = dorsal_features(pts, depth, seg, INTR)
    assert f["volume_full"] == pytest.approx(vol_true, rel=0.10)


def test_regressor_learns_allometric_relation():
    rng = np.random.default_rng(0)
    n = 200
    vol = rng.uniform(0.4, 1.2, n)
    X = pd.DataFrame({"volume_body": vol, "area_body": vol ** (2 / 3) * 1.1, "length_body": vol ** (1 / 3) * 1.5,
                      "width_p50": vol ** (1 / 3) * 0.5, "width_p90": vol ** (1 / 3) * 0.55,
                      "height_mean": vol ** (1 / 3) * 0.6, "height_p90": vol ** (1 / 3) * 0.7,
                      "hull_area_body": vol ** (2 / 3) * 1.2, "eq_cyl_radius": vol ** (1 / 3) * 0.3,
                      "vol_over_area": vol ** (1 / 3) * 0.9})
    y = 620 * vol * np.exp(rng.normal(0, 0.02, n))  # peso ∝ volume, 2% de ruído
    for kind in ["linear", "pls", "gbm"]:
        m = WeightRegressor(kind=kind).fit(X.iloc[:150], y[:150])
        p = m.predict(X.iloc[150:])
        assert regression_metrics(y[150:], p)["mape_pct"] < 4.0, kind


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    y = rng.uniform(300, 600, 2000)
    p = y * np.exp(rng.normal(0, 0.04, 2000))
    ci = ConformalInterval(alpha=0.1).fit(y[:1000], p[:1000])
    cov = ci.coverage(y[1000:], p[1000:])
    assert 0.86 <= cov <= 0.94


def test_herd_calibration_removes_bias():
    rng = np.random.default_rng(0)
    y = rng.uniform(300, 600, 30)
    p = y * 1.06 * np.exp(rng.normal(0, 0.02, 30))  # modelo global superestima 6%
    cal = HerdCalibrator("fazenda_x").fit(y, p)
    assert abs(regression_metrics(y, cal.apply(p))["bias_pct"]) < 1.0
    # com 3 pares o encolhimento segura o ajuste
    cal3 = HerdCalibrator("fazenda_y").fit(y[:3], p[:3])
    assert abs(cal3.a) < abs(np.log(1 / 1.06))


def test_metrics_and_aggregation():
    y = np.array([400.0, 500.0]); p = np.array([410.0, 490.0])
    m = regression_metrics(y, p)
    assert m["mape_pct"] == pytest.approx(2.25)
    assert bland_altman(y, p)["bias_kg"] == 0.0
    agg = aggregate_frames(np.array([500, 505, 498, 700]), np.array([1, 1, 1, 0]))
    assert agg["weight"] == 500 and agg["n_frames"] == 3 and agg["reliable"]


def test_silhouette_scale():
    m = np.zeros((200, 300), bool); m[50:150, 40:260] = True  # 220x100 px
    f = silhouette_features(m, px_per_m=100.0)
    assert f["sil_area"] == pytest.approx(2.2, rel=0.01)
