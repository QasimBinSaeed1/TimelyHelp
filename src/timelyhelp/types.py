"""Public, serializable interfaces. Observable records never contain simulator identities."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Candidate:
    source: str = "none"
    bridge: str = "cautious"
    delay: float = 0.0

    def __post_init__(self):
        if self.source not in ("none", "infrastructure", "supervisor"):
            raise ValueError(f"Unknown source {self.source}")
        if self.bridge not in ("cautious", "hold"):
            raise ValueError(f"Unknown bridge {self.bridge}")
        if not np.isfinite(self.delay) or self.delay < 0:
            raise ValueError("Delay must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls(source=value.source, bridge=value.bridge, delay=value.delay)


@dataclass
class Control:
    linear: float = 0.0
    angular: float = 0.0


@dataclass
class SimConfig:
    controller: str = "local"
    dt: float = 0.1
    horizon: float = 18.0
    commitment: float = 0.5
    robot_radius: float = 0.32
    max_speed: float = 1.0
    cautious_speed: float = 0.5
    acceleration: float = 1.2
    braking: float = 1.8
    sensor_range: float = 7.0
    camera_fov: float = 2.2
    camera_width: int = 64
    camera_height: int = 12
    track_timeout: float = 1.5
    max_reply_age: float = 2.0
    request_expiry: float = 6.0
    fusion_time: float = 0.1
    max_queries: int = 8
    near_miss_distance: float = 0.3
    history_length: int = 8
    collision_weight: float = 10.0
    failure_weight: float = 2.0
    time_weight: float = 0.1
    query_cost: float = 0.03
    detector_noise: float = 0.025
    source_latency: dict[str, list[float]] = field(
        default_factory=lambda: {"infrastructure": [0.2, 0.3, 0.5], "supervisor": [0.8, 1.5, 2.5]}
    )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None):
        values = values or {}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown simulation options: {sorted(unknown)}")
        obj = cls(**values)
        if obj.dt <= 0 or obj.horizon <= 0 or obj.commitment <= 0 or obj.fusion_time < 0:
            raise ValueError("Invalid timing configuration")
        return obj

    def to_dict(self):
        return asdict(self)


@dataclass
class Track:
    """A detector-assigned identity is internal association only, never an actor ID."""

    position: np.ndarray
    velocity: np.ndarray
    covariance: float
    capture_time: float
    kind: str
    provenance: str = "onboard"
    intent: str = "unknown"

    def public(self):
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "covariance": float(self.covariance),
            "capture_time": float(self.capture_time),
            "kind": self.kind,
            "provenance": self.provenance,
            "intent": self.intent,
        }


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    capture_time: float
    dropped: bool = False
    xy_map: np.ndarray | None = None


@dataclass
class Message:
    request_id: int
    source: str
    question: str
    capture_time: float
    arrival_time: float
    expires_at: float
    payload: dict[str, Any]
    bytes: int
    cancelled: bool = False
    admitted: bool = False
    reason: str = "pending"

    def public(self):
        return {
            "request_id": self.request_id,
            "source": self.source,
            "question": self.question,
            "capture_time": self.capture_time,
            "arrival_time": self.arrival_time,
            "payload": self.payload,
            "admitted": self.admitted,
            "reason": self.reason,
        }
