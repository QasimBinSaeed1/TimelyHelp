"""Disjoint layout splits and composed disturbance protocol for both mobility settings."""

from dataclasses import dataclass, asdict
import numpy as np

SPLITS = ("train", "development", "calibration", "test")
FAMILIES = ("worker_crossing", "cart_junction", "ambiguous_intent")
DISTURBANCES = (
    "occlusion",
    "crowd",
    "infrastructure_latency",
    "supervisor_latency",
    "no_response",
    "inaccurate_answer",
    "dropout",
    "calibration_error",
    "packet_loss",
    "stale_capture",
    "source_occlusion",
    "payload_dynamics",
)
# Combinations are reserved for test; constituent single disturbances occur in train/dev/calibration.
WITHHELD_COMPOSITIONS = (
    ("occlusion", "infrastructure_latency"),
    ("dropout", "packet_loss"),
    ("crowd", "inaccurate_answer", "payload_dynamics"),
    ("source_occlusion", "stale_capture", "supervisor_latency"),
)


@dataclass
class Scenario:
    layout_id: str
    split: str
    setting: str
    family: str
    disturbances: tuple[str, ...] = ()
    variant: int = 0

    def __post_init__(self):
        if self.split not in SPLITS or self.family not in FAMILIES:
            raise ValueError("Unknown split or event family")
        if self.setting not in ("worksite", "sidewalk"):
            raise ValueError("Unknown mobility setting")
        if set(self.disturbances) - set(DISTURBANCES):
            raise ValueError("Unknown disturbance")
        if len(self.disturbances) > 1 and self.split != "test":
            raise ValueError("Disturbance compositions are withheld for test only")

    @property
    def layout(self):
        return self.layout_id

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["disturbances"] = tuple(d.get("disturbances", ()))
        return cls(**d)

    def geometry(self):
        """Geometries differ by layout, not just rollout random seed."""
        group = SPLITS.index(self.split)
        crossing = 4.8 + group * 0.28 + self.variant * 0.35
        goal_x = 10.5 + 0.3 * group
        side = -1 if (self.variant + group) % 2 else 1
        halfwidth = 2.6 if self.setting == "worksite" else 2.8
        walls = [
            (0, -halfwidth - 0.3, goal_x + 1, -halfwidth),
            (0, halfwidth, goal_x + 1, halfwidth + 0.3),
        ]
        width = 1.7 if "occlusion" in self.disturbances else 1.15
        # The material stack / parked loading-area van occludes the approach.
        occluder = (crossing - width, side * 0.7, crossing - 0.4, side * 2.2)
        occluder = (
            occluder[0],
            min(occluder[1], occluder[3]),
            occluder[2],
            max(occluder[1], occluder[3]),
        )
        # Split-specific geometry includes different junction obstacle placements.
        extra = (
            [(7.5 + group * 0.2, -side * 2.2, 8.0 + group * 0.2, -side * 1.4)] if group % 2 else []
        )
        extra = [(a, min(b, d), c, max(b, d)) for a, b, c, d in extra]
        return {
            "rectangles": walls + [occluder] + extra,
            "start": [0.8, 0.0, 0.0],
            "goal": [goal_x, 0.0],
            "crossings": [[crossing, 0.0]],
            "camera_pose": [crossing + 0.1, -side * 2.3, side * np.pi / 2],
            "side": side,
            "crossing": crossing,
        }


def make_scenario(
    split="train", setting="worksite", family="worker_crossing", variant=0, disturbances=()
):
    return Scenario(
        f"{setting}_{split}_{variant:02d}", split, setting, family, tuple(disturbances), variant
    )


def benchmark_scenarios(split="test", variants=3, compositions=True):
    disturbances = [()] + [(d,) for d in DISTURBANCES]
    if split == "test" and compositions:
        disturbances += list(WITHHELD_COMPOSITIONS)
    return [
        make_scenario(split, s, f, v, d)
        for s in ("worksite", "sidewalk")
        for f in FAMILIES
        for v in range(variants)
        for d in disturbances
    ]


# Names convenient for configuration and native adapters.
build_scenario = make_scenario
