"""Parâmetros de câmera e presets de sensores."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrínsecos pinhole em pixels. depth_scale converte a unidade do PNG para metros."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 0.001  # mm -> m
    name: str = "custom"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraIntrinsics":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


# Valores nominais publicados/medidos; recalibrar na instalação.
PRESETS: dict[str, CameraIntrinsics] = {
    # Kinect v2 (sensor ToF 512x424) - usado no CowDatabase2
    "kinect_v2": CameraIntrinsics(365.5, 365.5, 256.0, 212.0, 512, 424, 0.001, "kinect_v2"),
    # Orbbec Femto Bolt, modo depth 640x576 NFOV (nominal)
    "femto_bolt_nfov": CameraIntrinsics(504.0, 504.0, 320.0, 288.0, 640, 576, 0.001, "femto_bolt_nfov"),
    # Intel/RealSense D455 depth 848x480 (nominal)
    "realsense_d455": CameraIntrinsics(425.0, 425.0, 424.0, 240.0, 848, 480, 0.001, "realsense_d455"),
    # Luxonis OAK-D Pro depth 640x400 (nominal)
    "oak_d_pro": CameraIntrinsics(450.0, 450.0, 320.0, 200.0, 640, 400, 0.001, "oak_d_pro"),
}


def get_intrinsics(name_or_path: str) -> CameraIntrinsics:
    if name_or_path in PRESETS:
        return PRESETS[name_or_path]
    p = Path(name_or_path)
    if p.exists():
        return CameraIntrinsics.from_yaml(p)
    raise KeyError(f"Preset ou arquivo de intrínsecos não encontrado: {name_or_path}")
