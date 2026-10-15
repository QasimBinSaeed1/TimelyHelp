"""Empirical latency integration and conservative joint source/motion scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import numpy as np

from .data import Candidate
from .model import ConsequenceEnsemble


@dataclass(frozen=True)
class LatencyDistribution:
    delays: np.ndarray
    probabilities: np.ndarray
    response_probability: float = 1.0
    provenance: str = "measured accessible interactions"

    def __post_init__(self):
        d, p = np.asarray(self.delays, dtype=float), np.asarray(self.probabilities, dtype=float)
        if d.ndim != 1 or p.shape != d.shape or not len(d):
            raise ValueError("Latency quadrature needs nonempty one-dimensional arrays")
        if not np.isfinite(d).all() or (d < 0).any() or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid latency quadrature")
        if not np.isclose(p.sum(), 1) or not 0 <= self.response_probability <= 1:
            raise ValueError("Invalid latency probabilities")
        object.__setattr__(self, "delays", d)
        object.__setattr__(self, "probabilities", p)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        source: str | None = None,
        max_points: int = 32,
        prior_delay: float = 1.0,
    ) -> LatencyDistribution:
        """Use previous measured total delays; never read a future simulator draw."""
        accessible = metadata.get(source, {}) if source is not None else metadata
        samples = accessible.get("latency_samples", accessible.get("latencies", []))
        samples = np.asarray(samples, dtype=float)
        if not np.isfinite(samples).all() or (samples < 0).any():
            raise ValueError("Measured latencies must be finite and nonnegative")
        provenance = str(accessible.get("latency_provenance", "measured accessible interactions"))
        if "last_arrival" in accessible:
            provenance = (
                "advertised prior samples; no completed replies"
                if accessible["last_arrival"] is None
                else "advertised prior samples plus measured completed replies"
            )
        if not len(samples):
            samples = np.asarray([accessible.get("advertised_latency", prior_delay)], dtype=float)
            provenance = "advertised latency prior; no measured successful replies"
        values, counts = np.unique(samples, return_counts=True)
        if max_points <= 0:
            raise ValueError("Quadrature point budget must be positive")
        # Equal-mass bins preserve the empirical mean; prediction integration is
        # approximate when compression is requested for nonlinear curves.
        if len(values) > max_points:
            chunks = np.array_split(np.sort(samples), max_points)
            values = np.asarray([chunk.mean() for chunk in chunks])
            counts = np.asarray([len(chunk) for chunk in chunks])
            provenance += "; mean-preserving empirical bin compression"
        response = float(
            accessible.get("response_fraction", accessible.get("response_probability", 1))
        )
        return cls(values, counts / counts.sum(), response, provenance)


@dataclass
class SchedulerConfig:
    risk_limit: float = 0.1
    uncertainty_weight: float = 1.0
    query_cost_weight: float = 0.05
    source_costs: dict[str, float] = field(
        default_factory=lambda: {"infrastructure": 1.0, "supervisor": 3.0}
    )
    commitment_horizon: float = 0.5
    expiration: float = 5.0
    max_capture_age: float = 3.0
    minimum_information_value: float = 0.0
    no_uncertainty: bool = False
    no_hold: bool = False
    maximum_latency_points: int = 32
    prior_delays: dict[str, float] = field(
        default_factory=lambda: {"infrastructure": 0.5, "supervisor": 2.0}
    )
    packet_age_feature_indices: dict[str, int] = field(
        default_factory=lambda: {"infrastructure": 24, "supervisor": 25}
    )
    packet_age_feature_scale: float = 6.0

    def __post_init__(self):
        if not 0 <= self.risk_limit <= 1 or self.uncertainty_weight < 0:
            raise ValueError("Invalid scheduler risk or uncertainty settings")
        if min(self.commitment_horizon, self.expiration, self.max_capture_age) <= 0:
            raise ValueError("Commitment, expiration and freshness horizons must be positive")


@dataclass(frozen=True)
class Decision:
    source: str
    bridge: str
    delay: float
    net_value: float
    collision_upper: float
    feasible: bool
    candidate_scores: tuple[dict[str, Any], ...] = ()
    commitment_horizon: float = 0.5

    @property
    def candidate(self) -> Candidate:
        return Candidate(self.source, self.bridge, self.delay)


class LearnedScheduler:
    def __init__(self, model: ConsequenceEnsemble, config: SchedulerConfig | None = None):
        self.model = model
        self.config = config or SchedulerConfig()
        self.selection_calibration: dict[str, Any] | None = None

    def configure_thresholds(self, report: Mapping[str, Any]) -> None:
        """Apply thresholds chosen on independent closed-loop calibration episodes.

        The experiment orchestrator runs the actual controller on each declared
        threshold grid. Paired-label calibration alone cannot establish deployed
        selection risk, so this method requires closed-loop provenance.
        """
        if report.get("split") != "calibration" or report.get("evaluation") != "closed_loop":
            raise ValueError("Selection thresholds require independent closed-loop calibration")
        if int(report.get("episodes", 0)) <= 0:
            raise ValueError("Selection calibration has no episodes")
        allowed = {"risk_limit", "uncertainty_weight", "minimum_information_value"}
        thresholds = report.get("thresholds", {})
        if not thresholds or set(thresholds) - allowed:
            raise ValueError("Unknown or empty selection threshold configuration")
        updated = asdict(self.config) | dict(thresholds)
        self.config = SchedulerConfig(**updated)
        self.selection_calibration = dict(report)

    def _score(
        self, history: np.ndarray, source: str, bridge: str, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        config = self.config
        if source == "none":
            delays, weights, response = (
                np.asarray([config.commitment_horizon]),
                np.asarray([1.0]),
                0.0,
            )
            candidates = [Candidate("none", bridge, config.commitment_horizon)]
            provenance = "finite no-query commitment"
        else:
            latency_metadata = metadata
            if getattr(self.model.config, "omit_reliability_metadata", False):
                # Mask the scheduling path too: historical response fraction is
                # source reliability metadata even when the GRU cannot see it.
                latency_metadata = dict(metadata)
                latency_metadata[source] = {
                    k: v
                    for k, v in metadata.get(source, {}).items()
                    if k not in ("response_fraction", "response_probability", "reliability_prior")
                }
            distribution = LatencyDistribution.from_metadata(
                latency_metadata, source, config.maximum_latency_points, config.prior_delays[source]
            )
            delays, weights, response = (
                distribution.delays,
                distribution.probabilities,
                distribution.response_probability,
            )
            candidates = [
                Candidate(source, bridge, min(float(delay), config.expiration)) for delay in delays
            ]
            # Replies beyond expiration are never fused, even when successful.
            candidates = [
                Candidate(
                    "none"
                    if delay > config.expiration
                    and not getattr(self.model.config, "omit_latency", False)
                    else source,
                    bridge,
                    c.delay,
                )
                for delay, c in zip(delays, candidates)
            ]
            weights = weights * response
            if response < 1:
                candidates.append(Candidate("none", bridge, config.expiration))
                delays = np.append(delays, config.expiration)
                weights = np.append(weights, 1 - response)
            provenance = distribution.provenance
        prediction_candidates = candidates
        if self.model.config.without_W:
            # A is source-independent by design but the excluded 'none' embedding
            # is untrained. Evaluate A at a trained source descriptor instead.
            prediction_candidates = [
                Candidate("infrastructure", c.bridge, c.delay) if c.source == "none" else c
                for c in candidates
            ]
        predictions = self.model.predict(history, prediction_candidates)
        factor = 0.0 if config.no_uncertainty else config.uncertainty_weight
        info = np.asarray(predictions["info"]).copy()
        motion = np.asarray(predictions["motion"]).copy()
        info_uncertainty = np.asarray(predictions["info_uncertainty"]).copy()
        noquery = np.asarray([c.source == "none" for c in candidates])
        # No-answer information benefit is an exact intervention identity.
        info[noquery], info_uncertainty[noquery] = 0, 0
        branches = np.full(len(candidates), 2, dtype=int)
        if self.model.config.without_W:
            # Strict A/Q-only ablation: no W or W-derived control targets are used.
            # A risk is a declared fallback proxy, including for the holding arm.
            branches[noquery] = 0
            motion[:] = 0
        uncertainty = info_uncertainty + np.asarray(predictions["motion_uncertainty"])
        if self.model.config.without_W:
            uncertainty[noquery] = 0
        rows = np.arange(len(candidates))
        collision = predictions["collision"][rows, branches]
        collision_upper = np.minimum(
            1, collision + factor * predictions["collision_uncertainty"][rows, branches]
        )
        lower_info = float(np.dot(weights, info - factor * info_uncertainty))
        net = float(np.dot(weights, info + motion - factor * uncertainty))
        if source != "none":
            net -= config.query_cost_weight * config.source_costs.get(source, 1.0)
        risk = float(np.dot(weights, collision_upper))
        feasible = risk <= config.risk_limit and (
            source == "none" or lower_info > config.minimum_information_value
        )
        return {
            "source": source,
            "bridge": bridge,
            "delay": float(np.dot(weights, delays)),
            "net_value": net,
            "information_lower": lower_info,
            "collision_upper": risk,
            "completion": float(np.dot(weights, predictions["completion"][rows, branches])),
            "completion_time": float(np.dot(weights, predictions["time"][rows, branches])),
            "feasible": bool(feasible),
            "latency_provenance": provenance,
            "without_W_fallback": bool(self.model.config.without_W and source == "none"),
        }

    def select(
        self, history: np.ndarray, metadata: Mapping[str, Any], observation: Any = None
    ) -> Decision:
        bridges = ["cautious"] if self.config.no_hold else ["cautious", "hold"]
        sources = ["none"]
        for source in ("infrastructure", "supervisor"):
            source_metadata = metadata.get(source, {})
            available = bool(
                source_metadata.get("available", source_metadata.get("communication_status", True))
            )
            if available and metadata.get("remaining_query_budget", 1) > 0:
                sources.append(source)
        scores = [
            self._score(history, source, bridge, metadata)
            for source in sources
            for bridge in bridges
        ]
        feasible = [score for score in scores if score["feasible"]]
        if feasible:
            chosen = max(feasible, key=lambda row: (row["net_value"], row["source"] == "none"))
        else:
            # A transparent risk-minimizing no-query fallback is necessary when
            # no candidate passes empirical bounds. Never label it risk-feasible.
            controls = [score for score in scores if score["source"] == "none"]
            chosen = min(controls, key=lambda row: (row["collision_upper"], -row["net_value"]))
        return Decision(
            chosen["source"],
            chosen["bridge"],
            chosen["delay"],
            chosen["net_value"],
            chosen["collision_upper"],
            chosen["feasible"],
            tuple(scores),
            self.config.commitment_horizon,
        )

    def reassess_packet(
        self,
        history: np.ndarray,
        packet_metadata: Mapping[str, Any],
        now: float,
        source_metadata: Mapping[str, Any],
        latest_capture_time: float | None = None,
    ) -> dict[str, Any]:
        """Arrival gate sees timestamps/status and current evidence, never defects."""
        source = str(packet_metadata["source"])
        capture = float(packet_metadata["capture_time"])
        arrival = float(packet_metadata["arrival_time"])
        age = now - capture
        if capture > now or arrival > now or not np.isfinite([capture, arrival, now]).all():
            return {"accept": False, "reason": "invalid_timestamp"}
        if now - arrival > self.config.expiration or age > self.config.max_capture_age:
            return {"accept": False, "reason": "stale"}
        if latest_capture_time is not None and capture <= latest_capture_time:
            return {"accept": False, "reason": "superseded"}
        if packet_metadata.get("cancelled", False) or packet_metadata.get("expired", False):
            return {"accept": False, "reason": "inactive_request"}
        bridge = str(packet_metadata.get("bridge", "cautious"))
        current = np.asarray(history, dtype=np.float32).copy()
        index = self.config.packet_age_feature_indices.get(source)
        if index is not None and current.shape[-1] > index:
            # Packet headers are now observable. Age conditions the current
            # evidence model without disclosing source defects or payload truth.
            current[-1, index] = age / self.config.packet_age_feature_scale
        prediction = self.model.predict(current, [Candidate(source, bridge, 0.0)])
        factor = 0.0 if self.config.no_uncertainty else self.config.uncertainty_weight
        value = float(prediction["info"][0] - factor * prediction["info_uncertainty"][0])
        risk = min(
            1.0,
            float(
                prediction["collision"][0, 2] + factor * prediction["collision_uncertainty"][0, 2]
            ),
        )
        accept = value > self.config.minimum_information_value and risk <= self.config.risk_limit
        return {
            "accept": bool(accept),
            "reason": "predicted_useful" if accept else "current_evidence_not_useful",
            "information_lower": value,
            "collision_upper": risk,
            "capture_age": age,
        }
