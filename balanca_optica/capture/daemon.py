"""Daemon de captura do ponto fixo: detecta presença, grava clipes, estima peso, enfileira.

Estrutura gravada:
  <out>/sessions/<AAAA-MM-DD>/<clip_id>/depth_0000.png (uint16 mm) rgb_0000.jpg meta.json
  <out>/readings.csv        uma linha por clipe (estimativa, IC, qualidade, tag RFID)
  <out>/queue/<clip_id>     marcador de pendência para o uploader

Presença: fração de pixels da ROI com altura acima do piso > h_min. O piso é estimado
por RANSAC no primeiro frame e re-estimado a cada `floor_refresh_s` quando não há animal.
Funciona offline; o uploader sincroniza quando houver rede.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np

from ..io.depth import depth_to_points
from ..segmentation.depth_seg import FloorPlane, fit_floor_plane, segment_animal
from ..geometry.features import dorsal_features
from ..aggregate import aggregate_frames
from .sources import DepthFrame, DepthSource


@dataclass
class CaptureConfig:
    out_dir: str = "capture_out"
    site: str = "fazenda_serra_bebedouro"
    h_min: float = 0.35  # altura mínima acima do piso para contar como animal (m)
    presence_on: float = 0.04  # fração da ROI ocupada para iniciar clipe
    presence_off: float = 0.01  # fração para encerrar clipe
    min_clip_frames: int = 4
    max_clip_s: float = 20.0
    save_every: int = 1  # salva 1 a cada N frames do clipe
    floor_refresh_s: float = 600.0
    roi: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1 em pixels; None = tudo
    model_path: str | None = None  # WeightRegressor .joblib para estimativa local
    jpeg_quality: int = 85


@dataclass
class ClipState:
    clip_id: str
    started: float
    frames: list = field(default_factory=list)  # (DepthFrame, idx)
    n_seen: int = 0


class CaptureDaemon:
    def __init__(self, source: DepthSource, cfg: CaptureConfig, rfid_reader=None, on_reading=None):
        self.src, self.cfg = source, cfg
        self.rfid = rfid_reader  # objeto com .latest_tag(max_age_s) -> str|None
        self.on_reading = on_reading
        self.out = Path(cfg.out_dir)
        (self.out / "sessions").mkdir(parents=True, exist_ok=True)
        (self.out / "queue").mkdir(exist_ok=True)
        self.floor: FloorPlane | None = None
        self._floor_ts = 0.0
        self._clip: ClipState | None = None
        self.model = None
        if cfg.model_path:
            from ..model.regressor import WeightRegressor
            self.model = WeightRegressor.load(cfg.model_path)
        self._readings = self.out / "readings.csv"
        if not self._readings.exists():
            with open(self._readings, "w", newline="") as f:
                csv.writer(f).writerow(["clip_id", "site", "ts_start", "ts_end", "n_frames", "n_ok",
                                        "rfid", "weight_kg", "rel_mad", "reliable", "path"])

    # ---- piso e presença -------------------------------------------------
    def _roi(self, a: np.ndarray) -> np.ndarray:
        if self.cfg.roi is None:
            return a
        x0, y0, x1, y1 = self.cfg.roi
        return a[y0:y1, x0:x1]

    def _ensure_floor(self, pts: np.ndarray, valid: np.ndarray, ts: float, occupied: bool):
        if self.floor is None or (not occupied and ts - self._floor_ts > self.cfg.floor_refresh_s):
            try:
                self.floor = fit_floor_plane(pts, valid)
                self._floor_ts = ts
            except ValueError:
                pass

    def presence(self, frame: DepthFrame) -> float:
        valid = frame.depth_m > 0
        pts = depth_to_points(frame.depth_m, self.src.intrinsics)
        if self.floor is None:
            self._ensure_floor(pts, valid, frame.ts, occupied=False)
            if self.floor is None:
                return 0.0
        h = self.floor.height(pts)
        occ = self._roi(valid & (h > self.cfg.h_min) & (h < 2.2))
        frac = float(occ.mean())
        self._ensure_floor(pts, valid, frame.ts, occupied=frac > self.cfg.presence_off)
        return frac

    # ---- clipes ---------------------------------------------------------
    def _start_clip(self, ts: float):
        cid = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts)) + f"_{int((ts % 1) * 1000):03d}"
        self._clip = ClipState(cid, ts)

    def _finish_clip(self):
        clip = self._clip
        self._clip = None
        if clip is None or len(clip.frames) < self.cfg.min_clip_frames:
            return None
        day = time.strftime("%Y-%m-%d", time.localtime(clip.started))
        d = self.out / "sessions" / day / clip.clip_id
        d.mkdir(parents=True, exist_ok=True)
        preds, oks = [], []
        for i, (fr, _) in enumerate(clip.frames):
            mm = np.clip(fr.depth_m / self.src.intrinsics.depth_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(d / f"depth_{i:04d}.png"), mm)
            if fr.rgb is not None:
                cv2.imwrite(str(d / f"rgb_{i:04d}.jpg"), fr.rgb, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if self.model is not None:
                try:
                    pts = depth_to_points(fr.depth_m, self.src.intrinsics)
                    seg = segment_animal(pts, fr.depth_m > 0, floor=self.floor)
                    feats = dorsal_features(pts, fr.depth_m, seg, self.src.intrinsics)
                    import pandas as pd
                    preds.append(float(self.model.predict(pd.DataFrame([feats]))[0]))
                    oks.append(bool(seg.quality["ok"]))
                except Exception:  # noqa: BLE001
                    pass
        tag = self.rfid.latest_tag(max_age_s=30) if self.rfid else None
        agg = aggregate_frames(np.array(preds), np.array(oks)) if preds else {"weight": np.nan, "n_frames": 0, "rel_mad": np.nan, "reliable": False}
        meta = {"clip_id": clip.clip_id, "site": self.cfg.site, "ts_start": clip.started,
                "ts_end": clip.frames[-1][0].ts, "n_frames": len(clip.frames), "rfid": tag,
                "camera": self.src.intrinsics.to_dict(),
                "floor": {"normal": self.floor.normal.tolist(), "d": self.floor.d} if self.floor else None,
                "estimate": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in agg.items()},
                "config": asdict(self.cfg)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        (self.out / "queue" / clip.clip_id).write_text(str(d))
        with open(self._readings, "a", newline="") as f:
            csv.writer(f).writerow([clip.clip_id, self.cfg.site, clip.started, meta["ts_end"], len(clip.frames),
                                    int(sum(oks)), tag, None if np.isnan(agg["weight"]) else round(agg["weight"], 1),
                                    None if np.isnan(agg["rel_mad"]) else round(agg["rel_mad"], 4), agg["reliable"], str(d)])
        if self.on_reading:
            self.on_reading(meta)
        return meta

    def step(self, frame: DepthFrame):
        frac = self.presence(frame)
        if self._clip is None:
            if frac >= self.cfg.presence_on:
                self._start_clip(frame.ts)
        if self._clip is not None:
            self._clip.n_seen += 1
            if frac > self.cfg.presence_off and self._clip.n_seen % self.cfg.save_every == 0:
                self._clip.frames.append((frame, self._clip.n_seen))
            ended = frac <= self.cfg.presence_off or (frame.ts - self._clip.started) > self.cfg.max_clip_s
            if ended:
                return self._finish_clip()
        return None

    def run(self, max_frames: int | None = None):
        n = 0
        try:
            for fr in self.src.frames():
                self.step(fr)
                n += 1
                if max_frames and n >= max_frames:
                    break
        finally:
            if self._clip is not None:
                self._finish_clip()
            self.src.close()
