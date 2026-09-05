"""Fontes de profundidade para o ponto fixo, com a mesma interface.

Kit de investimento mínimo (Fase 0): Kinect v2 usado (~R$ 400 a 600) + adaptador USB 3
+ notebook antigo -> `kinect2`. Alternativas: Kinect v1 / Orbbec Astra via OpenNI2
(`openni2`, roda até em Raspberry Pi 4), Orbbec Femto/Gemini (`orbbec`), RealSense (`realsense`).
`replay:<pasta>` reproduz PNGs gravados (testes e desenvolvimento sem hardware).

URI: "kinect2", "openni2", "orbbec", "realsense[:serial]", "replay:/caminho[:fps]"
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from ..config import CameraIntrinsics, PRESETS


@dataclass
class DepthFrame:
    depth_m: np.ndarray  # (H,W) float32 metros, 0 = inválido
    rgb: np.ndarray | None  # BGR alinhado à profundidade (ou None)
    ts: float  # epoch seconds


class DepthSource(Protocol):
    intrinsics: CameraIntrinsics

    def frames(self) -> Iterator[DepthFrame]: ...
    def close(self) -> None: ...


class ReplaySource:
    """Reproduz uma pasta com depth_*.png (uint16 mm) e rgb_*.png/jpg opcionais, em loop ou não."""

    def __init__(self, folder: str | Path, intrinsics: CameraIntrinsics = PRESETS["kinect_v2"],
                 fps: float = 5.0, loop: bool = False):
        self.folder = Path(folder)
        self.intrinsics = intrinsics
        self.fps, self.loop = fps, loop
        self.files = sorted(p for p in self.folder.rglob("*.png") if "depth" in p.name.lower())
        if not self.files:
            raise FileNotFoundError(f"nenhum depth*.png em {folder}")

    def frames(self) -> Iterator[DepthFrame]:
        while True:
            for f in self.files:
                raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                depth = raw.astype(np.float32) * self.intrinsics.depth_scale
                depth[raw == 0] = 0
                rgb = None
                for cand in (f.with_name(f.name.replace("depth", "rgb")),
                             f.with_name(f.name.replace("depth", "rgb")).with_suffix(".jpg")):
                    if cand.exists():
                        rgb = cv2.imread(str(cand))
                        break
                yield DepthFrame(depth, rgb, time.time())
                if self.fps:
                    time.sleep(1.0 / self.fps)
            if not self.loop:
                return

    def close(self) -> None:  # noqa: D401
        pass


class Kinect2Source:
    """Kinect v2 via pylibfreenect2 (libfreenect2). USB 3 obrigatório."""

    def __init__(self):
        try:
            from pylibfreenect2 import (Freenect2, FrameType, SyncMultiFrameListener,  # type: ignore
                                        createConsoleLogger, setGlobalLogger, LoggerLevel)
        except ImportError as e:  # pragma: no cover
            raise ImportError("instale libfreenect2 + pylibfreenect2 (ver docs/hardware.md)") from e
        setGlobalLogger(createConsoleLogger(LoggerLevel.Warning))
        self._fn = Freenect2()
        if self._fn.enumerateDevices() == 0:
            raise RuntimeError("Kinect v2 não encontrado")
        self._dev = self._fn.openDefaultDevice()
        self._listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self._dev.setColorFrameListener(self._listener)
        self._dev.setIrAndDepthFrameListener(self._listener)
        self._dev.start()
        self._FrameType = FrameType
        ip = self._dev.getIrCameraParams()
        self.intrinsics = CameraIntrinsics(ip.fx, ip.fy, ip.cx, ip.cy, 512, 424, 0.001, "kinect_v2_dev")

    def frames(self) -> Iterator[DepthFrame]:
        while True:
            fr = self._listener.waitForNewFrame()
            depth = fr["depth"].asarray().astype(np.float32) / 1000.0  # já em mm
            color = fr["color"].asarray()[:, :, :3]
            self._listener.release(fr)
            yield DepthFrame(depth, np.ascontiguousarray(color), time.time())

    def close(self) -> None:
        self._dev.stop(); self._dev.close()


class OpenNI2Source:
    """Kinect v1 / Asus Xtion / Orbbec Astra via OpenNI2 (pacote `primesense`)."""

    def __init__(self, redist: str | None = None):
        try:
            from primesense import openni2  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install primesense e aponte OPENNI2_REDIST") from e
        openni2.initialize(redist) if redist else openni2.initialize()
        self._dev = openni2.Device.open_any()
        self._stream = self._dev.create_depth_stream()
        self._stream.start()
        self._openni2 = openni2
        # Kinect v1: 640x480, fx≈fy≈525 nominal
        self.intrinsics = CameraIntrinsics(525.0, 525.0, 320.0, 240.0, 640, 480, 0.001, "openni2")

    def frames(self) -> Iterator[DepthFrame]:
        while True:
            f = self._stream.read_frame()
            buf = np.frombuffer(f.get_buffer_as_uint16(), dtype=np.uint16).reshape(f.height, f.width)
            yield DepthFrame(buf.astype(np.float32) / 1000.0, None, time.time())

    def close(self) -> None:
        self._stream.stop(); self._openni2.unload()


class RealSenseSource:
    def __init__(self, serial: str | None = None, w: int = 848, h: int = 480, fps: int = 15):
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install pyrealsense2") from e
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        prof = self._pipe.start(cfg)
        ds = prof.get_device().first_depth_sensor().get_depth_scale()
        i = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height, ds, "realsense_dev")
        self._align = rs.align(rs.stream.depth)

    def frames(self) -> Iterator[DepthFrame]:
        while True:
            fs = self._align.process(self._pipe.wait_for_frames())
            d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.intrinsics.depth_scale
            c = np.asanyarray(fs.get_color_frame().get_data())
            yield DepthFrame(d, c, time.time())

    def close(self) -> None:
        self._pipe.stop()


class OrbbecSource:
    """Orbbec Femto Bolt / Gemini via pyorbbecsdk."""

    def __init__(self):
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install pyorbbecsdk") from e
        self._pipe = Pipeline()
        cfg = Config()
        prof = self._pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        cfg.enable_stream(prof)
        self._pipe.start(cfg)
        intr = prof.get_intrinsic()
        self.intrinsics = CameraIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy, prof.get_width(),
                                           prof.get_height(), 0.001, "orbbec_dev")

    def frames(self) -> Iterator[DepthFrame]:
        while True:
            fs = self._pipe.wait_for_frames(1000)
            if fs is None:
                continue
            df = fs.get_depth_frame()
            if df is None:
                continue
            d = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
            yield DepthFrame(d.astype(np.float32) * df.get_depth_scale() / 1000.0, None, time.time())

    def close(self) -> None:
        self._pipe.stop()


def open_source(uri: str) -> DepthSource:
    kind, _, arg = uri.partition(":")
    if kind == "replay":
        folder, _, fps = arg.partition(":")
        return ReplaySource(folder, fps=float(fps) if fps else 5.0)
    if kind == "kinect2":
        return Kinect2Source()
    if kind == "openni2":
        return OpenNI2Source(arg or None)
    if kind == "realsense":
        return RealSenseSource(arg or None)
    if kind == "orbbec":
        return OrbbecSource()
    raise ValueError(f"fonte desconhecida: {uri}")
