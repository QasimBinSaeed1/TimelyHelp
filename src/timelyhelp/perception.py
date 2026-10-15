"""RGB-D rendering, color detector and observation-only nearest-neighbor tracker.

The reference camera uses horizontal pinhole rays; vertical pixels represent the
visible object silhouette. The detector consumes only RGB, depth and camera pose.
No object identities or world tracks cross this interface.
"""

from __future__ import annotations
import numpy as np
from .types import CameraFrame, Track

COLORS = {
    "static": np.array([125, 125, 125], dtype=np.uint8),
    "worker": np.array([230, 70, 45], dtype=np.uint8),
    "cart": np.array([35, 80, 220], dtype=np.uint8),
    "background": np.array([170, 200, 225], dtype=np.uint8),
}


def ray_rectangle(origin, directions, rect):
    lo = np.array(rect[:2])
    hi = np.array(rect[2:])
    eps = 1e-12
    inv = np.divide(
        1.0, directions, out=np.full_like(directions, np.inf), where=np.abs(directions) > eps
    )
    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    near = np.max(np.minimum(t1, t2), axis=1)
    far = np.min(np.maximum(t1, t2), axis=1)
    valid = far >= np.maximum(near, 0.0)
    return np.where(valid, np.maximum(near, 0.0), np.inf)


def ray_circle(origin, directions, center, radius):
    offset = origin - np.asarray(center)
    b = directions @ offset
    c = float(offset @ offset - radius**2)
    disc = b * b - c
    root = np.sqrt(np.maximum(disc, 0))
    a = -b - root
    z = -b + root
    return np.where((disc >= 0) & (z >= 0), np.where(a >= 0, a, z), np.inf)


def render_rgbd(
    pose, rectangles, actors, capture_time, width=64, height=12, fov=2.2, max_range=7.0
):
    """Render first ray hits; actors passed in here stay behind the sensor boundary."""
    pose = np.asarray(pose, float)
    # Image x locations are pinhole coordinates, not uniformly spaced angles.
    x = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), width)
    angles = pose[2] + np.arctan(x)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    depths = np.full(width, max_range)
    rgb = np.tile(COLORS["background"], (width, 1))
    for rectangle in rectangles:
        hits = ray_rectangle(pose[:2], directions, rectangle)
        vis = hits < depths
        depths[vis] = hits[vis]
        rgb[vis] = COLORS["static"]
    for actor in actors:
        hits = ray_circle(pose[:2], directions, actor["position"], actor["radius"])
        vis = hits < depths
        depths[vis] = hits[vis]
        rgb[vis] = COLORS[actor["kind"]]
    return CameraFrame(
        np.repeat(rgb[None, :, :], height, axis=0),
        np.repeat(depths[None, :], height, axis=0).astype(np.float32),
        pose.copy(),
        float(capture_time),
    )


class ColorDepthDetector:
    def __init__(self, fov=2.2, noise=0.025):
        self.fov = fov
        self.noise = noise

    def detect(self, frame, noise_draw=lambda i: 0.0):
        if frame.dropped:
            return []
        if frame.xy_map is not None:
            return self._detect_projected(frame, noise_draw)
        color = frame.rgb.mean(axis=0)
        depth = np.median(frame.depth, axis=0)
        w = len(depth)
        x = np.linspace(-np.tan(self.fov / 2), np.tan(self.fov / 2), w)
        result = []
        for kind, criterion in [
            ("worker", (color[:, 0] > 1.45 * color[:, 1]) & (color[:, 0] > 1.45 * color[:, 2])),
            ("cart", (color[:, 2] > 1.45 * color[:, 0]) & (color[:, 2] > 1.45 * color[:, 1])),
        ]:
            edges = np.diff(np.pad(criterion.astype(int), (1, 1)))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for start, end in zip(starts, ends):
                if end - start < 1:
                    continue
                middle = (start + end - 1) / 2
                angle = frame.pose[2] + np.arctan(np.interp(middle, np.arange(w), x))
                # Correct ray-hit depth to object center using the declared class radius.
                d = float(np.median(depth[start:end])) + (0.26 if kind == "worker" else 0.36)
                d += self.noise * noise_draw(len(result))
                position = frame.pose[:2] + d * np.array([np.cos(angle), np.sin(angle)])
                result.append(
                    {
                        "position": position,
                        "kind": kind,
                        "capture_time": frame.capture_time,
                        "covariance": float(self.noise**2 + 0.015 * d * d),
                    }
                )
        return result

    def _detect_projected(self, frame, noise_draw):
        from scipy.ndimage import label

        color = frame.rgb.astype(float)
        result = []
        criteria = [
            (
                "worker",
                (color[:, :, 0] > 1.45 * color[:, :, 1]) & (color[:, :, 0] > 1.45 * color[:, :, 2]),
            ),
            (
                "cart",
                (color[:, :, 2] > 1.45 * color[:, :, 0]) & (color[:, :, 2] > 1.45 * color[:, :, 1]),
            ),
        ]
        for kind, mask in criteria:
            components, count = label(mask)
            for component in range(1, count + 1):
                visible = (components == component) & np.isfinite(frame.xy_map).all(axis=2)
                if visible.sum() < 3:
                    continue
                points = frame.xy_map[visible]
                position = np.median(points, axis=0)
                ray = position - frame.pose[:2]
                distance = np.linalg.norm(ray)
                if distance > 1e-6:
                    position += (0.26 if kind == "worker" else 0.36) * ray / distance
                position += self.noise * noise_draw(len(result)) * np.array([1.0, 0.0])
                result.append(
                    {
                        "position": position,
                        "kind": kind,
                        "capture_time": frame.capture_time,
                        "covariance": float(self.noise**2 + 0.015 * distance**2),
                    }
                )
        return result


class ObservationTracker:
    def __init__(self, timeout=1.5):
        self.timeout = timeout
        self.tracks = []

    def update(self, detections, time, provenance="onboard"):
        tracks = [t for t in self.tracks if time - t.capture_time <= self.timeout]
        available = list(range(len(tracks)))
        for d in detections:
            if time - d["capture_time"] > self.timeout:
                continue
            choices = [i for i in available if tracks[i].kind == d["kind"]]
            matched = (
                min(choices, key=lambda i: np.linalg.norm(tracks[i].position - d["position"]))
                if choices
                else None
            )
            if (
                matched is not None
                and np.linalg.norm(tracks[matched].position - d["position"]) < 1.6
            ):
                t = tracks[matched]
                dt = d["capture_time"] - t.capture_time
                # Do not let asynchronous stale captures overwrite newer observed tracks.
                if dt >= 0:
                    if dt > 1e-6:
                        raw = (d["position"] - t.position) / dt
                        t.velocity = 0.55 * t.velocity + 0.45 * np.clip(raw, -2.5, 2.5)
                    if "velocity" in d:
                        t.velocity = np.clip(np.asarray(d["velocity"], float), -2.5, 2.5)
                    t.position = d["position"].copy()
                    t.capture_time = d["capture_time"]
                    t.covariance = d["covariance"]
                    t.provenance = provenance
                available.remove(matched)
            else:
                tracks.append(
                    Track(
                        d["position"].copy(),
                        np.asarray(d.get("velocity", [0.0, 0.0]), float),
                        d["covariance"],
                        d["capture_time"],
                        d["kind"],
                        provenance,
                    )
                )
        self.tracks = tracks
        return tracks
