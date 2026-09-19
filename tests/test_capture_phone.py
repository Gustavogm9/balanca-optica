import numpy as np, cv2, pandas as pd
from pathlib import Path
from balanca_optica.capture.sources import ReplaySource
from balanca_optica.capture.daemon import CaptureDaemon, CaptureConfig
from balanca_optica.capture.pairing import pair_clips_with_scale
from balanca_optica.geometry.silhouette import silhouette_features
from balanca_optica.phone.import_media import process_media, PhoneRecord

import sys; sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import synthetic_depth, INTR


def test_daemon_records_clip_and_pairs_with_scale(tmp_path):
    rep = tmp_path / "replay"; rep.mkdir()
    empty = np.full((INTR.height, INTR.width), 3200, np.uint16)
    body, _ = synthetic_depth()
    seq = [empty] * 2 + [body] * 6 + [empty] * 2
    for i, d in enumerate(seq):
        cv2.imwrite(str(rep / f"depth_{i:04d}.png"), (d * 1000).astype(np.uint16) if d.dtype != np.uint16 else d)
    src = ReplaySource(rep, intrinsics=INTR, fps=0)
    out = tmp_path / "cap"
    d = CaptureDaemon(src, CaptureConfig(out_dir=str(out), site="t"))
    d.run()
    r = pd.read_csv(out / "readings.csv")
    assert len(r) == 1 and r.n_frames.iloc[0] >= 4
    assert len(list((out / "queue").iterdir())) == 1
    clip_dir = Path(r.path.iloc[0]); assert (clip_dir / "meta.json").exists()
    # balança pesou 10 s depois do clipe
    scale = tmp_path / "scale.csv"
    pd.DataFrame({"ts": [r.ts_end.iloc[0] + 10], "weight_kg": [512.0], "animal_id": ["BR001"]}).to_csv(scale, index=False)
    pairs = pair_clips_with_scale(str(out / "readings.csv"), str(scale), window_s=60)
    assert len(pairs) == 1 and pairs.animal_id.iloc[0] == "BR001"


class _FakeSeg:
    def mask(self, img):
        m = np.zeros(img.shape[:2], bool); h, w = m.shape
        m[h//3:2*h//3, w//4:3*w//4] = True  # "corpo"
        m[2*h//3:int(0.9*h), w//4:w//4+w//20] = True  # "perna"
        return m


def test_phone_import_with_marker_scale(tmp_path):
    img = np.full((600, 800, 3), 128, np.uint8)
    f = tmp_path / "a.jpg"; cv2.imwrite(str(f), img)
    recs = [PhoneRecord("A1", 400.0, f, "lateral", marker_px=100, marker_m=0.5)]
    df = process_media(recs, _FakeSeg(), verbose=False)
    assert len(df) == 1 and df.scale_src.iloc[0] == "marker"
    # corpo 200x400 px a 200 px/m -> 1 x 2 m = 2 m² (mais a perna)
    assert 2.0 <= df.sil_body_area.iloc[0] <= 2.3
    assert df.sil_body_h.iloc[0] < df.sil_bbox_h.iloc[0]  # perna excluída do tronco


def test_aruco_marker_roundtrip_scale(tmp_path):
    from balanca_optica.phone.marker import generate_marker, detect_marker, DEFAULT_SIDE_M
    m = generate_marker(3, side_px=300, border_px=40)
    scene = np.full((900, 1200), 140, np.uint8)
    scene[200:200 + m.shape[0], 700:700 + m.shape[1]] = m
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    hit = detect_marker(scene)
    assert hit is not None and hit.marker_id == 3
    assert abs(hit.side_px - 300) < 4
    assert abs(hit.px_per_m - 300 / DEFAULT_SIDE_M) / (300 / DEFAULT_SIDE_M) < 0.02
    # integração no importador: escala vem do aruco mesmo sem distance/marker manual
    f = tmp_path / "a.jpg"; cv2.imwrite(str(f), scene)
    df = process_media([PhoneRecord("A2", 400.0, f, "lateral")], _FakeSeg(), verbose=False)
    assert df.scale_src.iloc[0] == "aruco" and df.marker_id.iloc[0] == 3
