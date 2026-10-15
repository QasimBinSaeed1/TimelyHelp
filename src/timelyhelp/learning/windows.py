"""Nonmonotone useful-arrival sets with explicit finite-horizon censoring."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence
import numpy as np

from .data import Candidate


@dataclass(frozen=True)
class UsefulWindow:
    intervals: tuple[tuple[float, float], ...]
    right_censored: bool
    probe_horizon: float
    positive: tuple[bool, ...]
    delays: tuple[float, ...]

    @property
    def connected_deadline(self) -> float | None:
        if len(self.intervals) != 1 or self.right_censored:
            return None
        return self.intervals[0][1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "connected_deadline": self.connected_deadline,
            "resolution": "linear interpolation between finite delay probes",
        }


def useful_window(
    delays: Sequence[float],
    information_lower: Sequence[float],
    collision_upper: Sequence[float],
    risk_limit: float = 0.1,
) -> UsefulWindow:
    """Interpolate each threshold crossing; retain holes and boundary censoring.

    The estimate concerns only the supplied controller/source/bridge and sampled
    horizon. A positive last probe is right-censored, not an infinite deadline.
    """
    d, benefit, risk = [
        np.asarray(value, dtype=float) for value in (delays, information_lower, collision_upper)
    ]
    if d.ndim != 1 or len(d) < 2 or benefit.shape != d.shape or risk.shape != d.shape:
        raise ValueError("At least two aligned delay probes are required")
    if (
        not np.isfinite(np.concatenate((d, benefit, risk))).all()
        or (d < 0).any()
        or (np.diff(d) <= 0).any()
    ):
        raise ValueError("Delay probes must be finite, nonnegative and strictly increasing")
    if not 0 <= risk_limit <= 1:
        raise ValueError("Invalid useful-window risk limit")
    good = (benefit > 0) & (risk <= risk_limit)
    intervals = []
    for i in range(len(d) - 1):
        # Piecewise-linear inequalities can yield a useful subinterval even if
        # both endpoints fail different conditions (e.g. improving risk but
        # decreasing utility). Solve the two inequalities separately.
        lo, hi = 0.0, 1.0
        for v0, v1, strict in (
            (benefit[i], benefit[i + 1], True),
            (risk_limit - risk[i], risk_limit - risk[i + 1], False),
        ):
            if v0 == v1:
                if v0 < 0 or (strict and v0 == 0):
                    lo, hi = 1.0, 0.0
                    break
            else:
                crossing = -v0 / (v1 - v0)
                if v1 > v0:
                    lo = max(lo, crossing)
                else:
                    hi = min(hi, crossing)
        if lo < hi:
            start = float(d[i] + max(0.0, lo) * (d[i + 1] - d[i]))
            end = float(d[i] + min(1.0, hi) * (d[i + 1] - d[i]))
            if start < end:
                if intervals and good[i] and np.isclose(intervals[-1][1], start, atol=1e-10):
                    intervals[-1] = (intervals[-1][0], end)
                else:
                    intervals.append((start, end))
    # The execution-risk inequality is non-strict. A probe that touches the
    # limit from above on both sides is an isolated feasible arrival time,
    # provided its information benefit is strictly positive. Preserve that
    # singleton rather than reporting a contradictory empty useful set.
    for delay, useful in zip(d, good):
        if useful and not any(start <= delay <= end for start, end in intervals):
            intervals.append((float(delay), float(delay)))
    intervals.sort()
    return UsefulWindow(
        tuple(intervals),
        bool(good[-1]),
        float(d[-1]),
        tuple(bool(v) for v in good),
        tuple(float(v) for v in d),
    )


def predict_useful_window(
    model,
    history: np.ndarray,
    source: str,
    bridge: str,
    delays: Sequence[float],
    risk_limit: float = 0.1,
    uncertainty_weight: float = 1.0,
) -> UsefulWindow:
    prediction = model.predict(history, [Candidate(source, bridge, float(d)) for d in delays])
    lower = prediction["info"] - uncertainty_weight * prediction["info_uncertainty"]
    upper = np.minimum(
        1,
        prediction["collision"][:, 2]
        + uncertainty_weight * prediction["collision_uncertainty"][:, 2],
    )
    return useful_window(delays, lower, upper, risk_limit)
