"""Matched, observation-based assistance baselines and a separate diagnostic oracle.

The rules adapt confidence, failure monitoring and planning relevance to this
interface; they are not reproductions of unrelated original benchmark systems.
The strongest comparator is a genuine observation-conditioned Monte Carlo VOI
planner. Its generative crossing prior is explicit and can be calibrated using
development data. It never reads an environment, snapshot, hidden actor state,
future trajectory, simulator identity, or a pending message's unseen payload.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.stats import beta

from .control import EvidenceAdapter, make_planner, wrap
from .perception import ColorDepthDetector, ObservationTracker, render_rgbd
from .types import Candidate, Message, SimConfig, Track


@dataclass
class BaselineConfig:
    confidence_threshold: float = 0.65
    failure_threshold: float = 0.15
    risk_limit: float = 0.1
    conservative_z: float = 1.0
    query_cost: float = 0.03
    monte_carlo_samples: int = 8
    latency_support: int = 3
    hidden_occupancy_prior: float = 0.25
    hidden_uncertainty_weight: float = 0.5
    crossing_intent_prior: float = 0.65
    velocity_std: float = 0.2
    sim: SimConfig = field(default_factory=SimConfig)

    @classmethod
    def from_dict(cls, config: dict | None):
        config = deepcopy(config or {})
        if isinstance(config.get("sim"), dict):
            config["sim"] = SimConfig.from_dict(config["sim"])
        result = cls(**config)
        if result.monte_carlo_samples < 2 or result.latency_support < 1:
            raise ValueError("Monte Carlo requires >=2 samples and positive latency support")
        return result


@dataclass
class Decision:
    source: str = "none"
    bridge: str = "cautious"
    delay: float = 0.0
    net_value: float = 0.0
    collision_upper: float = 0.0
    feasible: bool = True
    candidate_scores: list[dict] = field(default_factory=list)
    deadline: float | None = None


def _latencies(metadata: dict, support: int = 3) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(metadata.get("latency_samples", metadata.get("latencies", [])), dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if not len(values):
        return np.array([]), np.array([])
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) <= support:
        return unique, counts / counts.sum()
    # Deterministic empirical quantile strata, retaining their actual mass.
    bins = np.array_split(np.sort(values), support)
    return np.asarray([b.mean() for b in bins]), np.asarray([len(b) / len(values) for b in bins])


def _available(metadata: dict) -> bool:
    return metadata.get("status", "available") in ("available", "pending", "online") and bool(
        metadata.get("latency_samples", metadata.get("latencies", []))
    )


def _observation_required(observation):
    if observation is None:
        raise ValueError("Observation-based baselines require the public observation argument")
    return observation


def _confidence(observation: dict) -> float:
    tracks = observation.get("tracks", [])
    variance = float(np.mean([track["covariance"] for track in tracks])) if tracks else 1.0
    return float(
        np.clip(
            1.0 / (1.0 + variance) * (1.0 - observation.get("blind_zone_uncertainty", 0.0)), 0, 1
        )
    )


def _lead_time(observation: dict, config: SimConfig) -> float:
    pose = np.asarray(observation["pose"])
    direction = np.array([np.cos(pose[2]), np.sin(pose[2])])
    ahead = [
        float((np.asarray(crossing) - pose[:2]) @ direction)
        for crossing in observation.get("crossings", [])
    ]
    ahead = [value for value in ahead if value > 0]
    distance = min(ahead, default=observation.get("depth_min", config.sensor_range))
    return max(
        0.0,
        (distance - config.robot_radius)
        / max(0.1, observation.get("velocity", 0.0), config.cautious_speed),
    )


def _source_score(metadata: dict, lead: float) -> tuple[str, float]:
    scored = []
    for source, information in metadata.items():
        if source not in ("infrastructure", "supervisor"):
            continue
        if not _available(information):
            continue
        delays, weights = _latencies(information)
        useful_mass = float(weights[delays <= lead].sum())
        score = (
            float(information.get("coverage", 0.5))
            * float(information.get("response_fraction", 1.0))
            * float(information.get("reliability_prior", 0.5))
            * useful_mass
        )
        scored.append((score, source, float(delays @ weights)))
    if not scored:
        return "none", 0.0
    _, source, delay = max(scored)
    return source, delay


class RuleBaseline:
    """Rule adaptations with common budget, planner and evidence fusion."""

    def __init__(self, name: str, config: BaselineConfig | None = None):
        self.name = name
        self.config = config or BaselineConfig()

    def reset(self, seed: int = 0):
        self.seed = seed

    def select(self, history, metadata, observation=None) -> Decision:
        o = _observation_required(observation)
        c = self.config
        if self.name == "no_query_nominal":
            return Decision(bridge="nominal")
        if self.name == "no_query_cautious":
            return Decision(bridge="cautious")
        lead = _lead_time(o, c.sim)
        source, delay = _source_score(metadata, lead)
        if o.get("remaining_queries", 0) <= 0:
            return Decision(bridge="cautious")
        if self.name == "always_query":
            source, delay = _source_score(metadata, float("inf"))
            return Decision(source, "cautious", delay)
        confidence = _confidence(o)
        if self.name == "confidence_triggered":
            return Decision(
                source if confidence < c.confidence_threshold else "none", "cautious", delay
            )
        planner = make_planner(c.sim)
        planner.plan(o, "nominal")
        failure_risk = max(planner.last_risk, 0.5 * o.get("blind_zone_uncertainty", 0.0))
        # Observable lack of progress is a Sentinel-style monitor adaptation.
        history = np.asarray(history)
        if history.ndim == 2 and len(history) > 1:
            progress = float(np.linalg.norm(history[-1, 1:3] - history[0, 1:3]))
            failure_risk = max(
                failure_risk, 0.3 if progress < 0.001 and o["time"] > c.sim.commitment else 0.0
            )
        if self.name == "failure_monitor_score":
            asks = failure_risk >= c.failure_threshold and source != "none" and delay <= lead
            return Decision(
                source if asks else "none", "hold" if failure_risk > 0.5 else "cautious", delay
            )
        if self.name == "planning_aware":
            # Timestamp-compensated tracks supply path conflict, not source identity.
            pose = np.asarray(o["pose"])
            direction = np.array([math.cos(pose[2]), math.sin(pose[2])])
            relevant = float(o.get("blind_zone_uncertainty", 0.0))
            for track in o.get("tracks", []):
                age = max(0.0, o["time"] - track["capture_time"])
                future = np.asarray(track["position"]) + np.asarray(track["velocity"]) * (
                    age + delay
                )
                rel = future - pose[:2]
                along = rel @ direction
                lateral = abs(rel @ np.array([-direction[1], direction[0]]))
                if 0 < along < c.sim.sensor_range:
                    relevant = max(
                        relevant, float(np.exp(-lateral * lateral / (1.0 + track["covariance"])))
                    )
            asks = relevant >= 0.25 and source != "none" and delay <= lead
            return Decision(source if asks else "none", "cautious", delay)
        raise ValueError(f"Unknown rule baseline {self.name}")


class FailureLeadTimeBaseline:
    """Calibrated nominal-outcome forecast plus an observable conflict lead.

    This comparator uses only A collision/completion heads from the frozen
    consequence model. It never estimates query benefit or consults hidden state.
    Lead time is an explicit constant-velocity conflict/crossing estimate, rather
    than an invented trained time-to-failure target.
    """

    name = "failure_lead_time"

    def __init__(self, predictor, config=None, provenance=None):
        if getattr(predictor, "calibration", None) is None:
            raise ValueError(
                "Failure probability baseline requires a held-out calibrated predictor"
            )
        self.predictor = predictor
        self.config = config or BaselineConfig()
        self.predictor_provenance = deepcopy(provenance or {})

    def reset(self, seed=0):
        self.seed = seed

    def select(self, history, metadata, observation=None):
        observation = _observation_required(observation)
        config = self.config
        prediction = self.predictor.predict(
            history, [Candidate("none", "cautious", config.sim.commitment)]
        )
        collision = float(prediction["collision"][0, 0])
        completion = float(prediction["completion"][0, 0])
        collision_upper = float(
            np.clip(
                collision + config.conservative_z * prediction["collision_uncertainty"][0, 0],
                0.0,
                1.0,
            )
        )
        completion_lower = float(
            np.clip(
                completion - config.conservative_z * prediction["completion_uncertainty"][0, 0],
                0.0,
                1.0,
            )
        )
        # Collision is a mission failure. The max also conservatively handles
        # finite-model incoherence between separately calibrated Bernoulli heads.
        failure_probability = max(collision, 1.0 - completion)
        failure_upper = max(collision_upper, 1.0 - completion_lower)
        lead = min(_lead_time(observation, config.sim), float(observation["remaining_time"]))
        pose = np.asarray(observation["pose"])
        robot_velocity = float(observation["velocity"]) * np.array(
            [np.cos(pose[2]), np.sin(pose[2])]
        )
        for track in observation.get("tracks", []):
            age = max(0.0, observation["time"] - track["capture_time"])
            relative_position = (
                np.asarray(track["position"]) + age * np.asarray(track["velocity"]) - pose[:2]
            )
            relative_velocity = np.asarray(track["velocity"]) - robot_velocity
            radius = (
                config.sim.robot_radius
                + (0.26 if track["kind"] == "worker" else 0.36)
                + np.sqrt(max(0.0, track["covariance"]))
            )
            a = float(relative_velocity @ relative_velocity)
            b = float(2.0 * relative_position @ relative_velocity)
            c = float(relative_position @ relative_position - radius * radius)
            discriminant = b * b - 4.0 * a * c
            if c <= 0:
                lead = 0.0
            elif a > 1e-9 and discriminant >= 0:
                first_contact = (-b - np.sqrt(discriminant)) / (2.0 * a)
                if first_contact >= 0:
                    lead = min(lead, float(first_contact))
        source, delay = _source_score(metadata, lead)
        asks = (
            observation.get("remaining_queries", 0) > 0
            and failure_upper >= config.failure_threshold
            and source != "none"
            and delay <= lead
        )
        score = {
            "source": source,
            "failure_probability": failure_probability,
            "failure_upper": failure_upper,
            "collision_probability": collision,
            "completion_probability": completion,
            "collision_upper": collision_upper,
            "predicted_lead_time": lead,
            "expected_reply_delay": delay,
            "probability_semantics": "calibrated nominal A mission-failure forecast",
            "lead_time_semantics": "public crossing and covariance-inflated constant-velocity conflict estimate",
        }
        return Decision(
            source if asks else "none",
            "hold" if collision_upper > 0.5 else "cautious",
            delay,
            collision_upper=collision_upper,
            feasible=collision_upper <= config.risk_limit,
            candidate_scores=[score],
        )


class _BeliefRollout:
    """A sampled synthetic world; initialized exclusively from public evidence."""

    def __init__(self, observation, actors, config: SimConfig, noise: np.ndarray):
        self.initial = deepcopy(observation)
        self.pose = np.asarray(observation["pose"], dtype=float).copy()
        self.velocity = float(observation["velocity"])
        self.goal = np.asarray(observation["goal"], dtype=float)
        self.time = float(observation["time"])
        self.actors = deepcopy(actors)
        self.config = config
        self.noise = noise
        self.rectangles = deepcopy(observation.get("map_obstacles", []))
        self.tracker = ObservationTracker(config.track_timeout)
        for track in observation.get("tracks", []):
            self.tracker.tracks.append(
                Track(
                    np.asarray(track["position"], float),
                    np.asarray(track["velocity"], float),
                    float(track["covariance"]),
                    float(track["capture_time"]),
                    track["kind"],
                    track.get("provenance", "onboard"),
                )
            )
        self.evidence = EvidenceAdapter(config)
        self.evidence.intent_beliefs = deepcopy(observation.get("intent_beliefs", []))
        self.planner = make_planner(config)
        self.detector = ColorDepthDetector(config.camera_fov, config.detector_noise)
        self.frame = None
        self.completed = False
        self.collision = False
        self.step_index = 0

    def observe(self):
        observation = deepcopy(self.initial)
        observation.update(
            pose=self.pose.tolist(),
            velocity=self.velocity,
            time=self.time,
            tracks=[track.public() for track in self.tracker.tracks],
            intent_beliefs=self.evidence.current_intents(self.time),
        )
        if self.frame is not None:
            middle = self.frame.depth[
                :, int(self.config.camera_width * 0.42) : int(self.config.camera_width * 0.58)
            ]
            observation["depth_min"] = float(np.min(middle))
            observation["depth_left"] = float(
                np.median(self.frame.depth[:, : self.config.camera_width // 3])
            )
            observation["depth_right"] = float(
                np.median(self.frame.depth[:, 2 * self.config.camera_width // 3 :])
            )
        return observation

    def packet(self, source: str, metadata: dict, delay: float, random: np.ndarray):
        c = self.config
        info = metadata[source]
        capture_age = max(0.0, float(info.get("capture_age", 0.0)))
        capture_time = self.time - capture_age
        answer = "unknown"
        payload = {"answer": answer}
        reliability = float(info.get("reliability_prior", 0.5))
        responded = random[0] < float(info.get("response_fraction", 1.0))
        if source == "infrastructure":
            pose = info.get("camera_pose")
            if pose is not None and random[1] < float(info.get("coverage", 0.5)):
                source_tracker = ObservationTracker(c.track_timeout)
                # Temporal source tracks, synthesized from sampled belief
                # velocities, reproduce the sensor/tracker interface rather
                # than disclosing sampled velocities as oracle detections.
                for lag in (2 * c.dt + capture_age, c.dt + capture_age, capture_age):
                    historical = deepcopy(self.actors)
                    for actor in historical:
                        actor["position"] -= actor["velocity"] * lag
                    source_frame = render_rgbd(
                        pose,
                        self.rectangles,
                        historical,
                        self.time - lag,
                        c.camera_width,
                        c.camera_height,
                        c.camera_fov,
                        c.sensor_range,
                    )
                    source_tracker.update(
                        self.detector.detect(source_frame),
                        self.time - lag,
                        provenance="infrastructure",
                    )
                detected = [track.public() for track in source_tracker.tracks]
                if random[2] > reliability:
                    for detection in detected:
                        detection["position"] = (
                            np.asarray(detection["position"])
                            + np.array([0.0, 1.5 * (2 * random[3] - 1)])
                        ).tolist()
                payload = {"answer": "tracks" if detected else "unknown", "tracks": detected}
        else:
            crossing = self.initial.get("crossings", [[0.0, 0.0]])[0]
            eligible = [
                actor
                for actor in self.actors
                if actor["kind"] == "worker"
                and np.linalg.norm(actor["position"] - actor["velocity"] * capture_age - crossing)
                < 4.5
            ]
            if eligible and random[1] < float(info.get("coverage", 0.5)):
                answer = "cross" if eligible[0]["crosses"] else "not_cross"
                if random[2] > reliability:
                    answer = "not_cross" if answer == "cross" else "cross"
            payload = {"answer": answer, "crossing": crossing}
        return Message(
            1,
            source,
            "local_tracks_at_crossing"
            if source == "infrastructure"
            else "will_you_cross_delivery_lane",
            capture_time,
            self.time + delay if responded else float("inf"),
            self.time + c.request_expiry,
            payload,
            0,
        )

    def run(self, bridge: str, duration: float, packet: Message | None):
        c = self.config
        initial_time = self.time
        remaining = float(self.initial["remaining_time"])
        ticks = int(np.ceil(remaining / c.dt))
        admitted = False
        for index in range(ticks):
            elapsed = self.time - initial_time
            if packet is not None and not admitted and self.time >= packet.arrival_time - 1e-9:
                self.evidence.admit(packet, self.tracker, self.time)
                admitted = True
            action = self.planner.plan(
                self.observe(), bridge if elapsed < duration - 1e-9 else "nominal"
            )
            limit = c.acceleration if action.linear > self.velocity else c.braking
            self.velocity += float(
                np.clip(action.linear - self.velocity, -limit * c.dt, limit * c.dt)
            )
            self.pose[2] = wrap(self.pose[2] + action.angular * c.dt)
            self.pose[:2] += (
                self.velocity * c.dt * np.array([np.cos(self.pose[2]), np.sin(self.pose[2])])
            )
            for actor_index, actor in enumerate(self.actors):
                delta = actor["destination"] - actor["position"]
                distance = np.linalg.norm(delta)
                desired = actor["speed"] * delta / max(distance, 1e-9)
                rel = actor["position"] - self.pose[:2]
                robot_distance = np.linalg.norm(rel)
                if robot_distance < 1.6:
                    desired += 0.25 * (1.6 - robot_distance) * rel / max(robot_distance, 1e-6)
                    desired *= 0.75 if self.velocity > 0.1 else 1.0
                jitter = 0.045 * self.noise[index, actor_index % self.noise.shape[1]]
                desired += jitter * np.array([-delta[1], delta[0]]) / max(distance, 1e-9)
                actor["velocity"] = desired if distance > 0.15 else np.zeros(2)
                actor["position"] += actor["velocity"] * c.dt
            self.time += c.dt
            self.frame = render_rgbd(
                self.pose,
                self.rectangles,
                self.actors,
                self.time,
                c.camera_width,
                c.camera_height,
                c.camera_fov,
                c.sensor_range,
            )
            detections = self.detector.detect(
                self.frame, lambda i: self.noise[index, (i + 4) % self.noise.shape[1]]
            )
            self.tracker.update(detections, self.time)
            gaps = [
                np.linalg.norm(actor["position"] - self.pose[:2]) - actor["radius"] - c.robot_radius
                for actor in self.actors
            ]
            gaps.extend(
                np.linalg.norm(self.pose[:2] - np.clip(self.pose[:2], r[:2], r[2:]))
                - c.robot_radius
                for r in self.rectangles
            )
            self.collision = bool(min(gaps, default=10.0) <= 0)
            self.completed = bool(
                not self.collision and np.linalg.norm(self.pose[:2] - self.goal) < 0.35
            )
            if self.completed or self.collision:
                break
        elapsed = self.time - initial_time
        cost = (
            c.collision_weight * self.collision
            + c.failure_weight * (not self.completed)
            + c.time_weight * self.time / c.horizon
        )
        return np.array([cost, float(self.collision), float(self.completed), elapsed])


class ModelBasedVOI:
    """Latency/reliability/bridge-conditioned Monte Carlo signed value of evidence."""

    name = "model_based_voi"

    def __init__(self, config: BaselineConfig | None = None, seed=0):
        self.config = config or BaselineConfig()
        self.reset(seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def _sample_actors(self, observation):
        c = self.config
        actors = []
        for track in observation.get("tracks", []):
            age = max(0.0, observation["time"] - track["capture_time"])
            velocity = np.asarray(track["velocity"]) + self.rng.normal(0, c.velocity_std, 2)
            position = np.asarray(track["position"]) + np.asarray(track["velocity"]) * age
            position = position + self.rng.normal(
                0, np.sqrt(max(1e-5, track["covariance"])) + 0.05 * age, 2
            )
            crossing = min(
                observation.get("crossings", [position]),
                key=lambda p: np.linalg.norm(np.asarray(p) - position),
            )
            intent = c.crossing_intent_prior
            for belief in observation.get("intent_beliefs", []):
                if np.linalg.norm(np.asarray(belief["crossing"]) - crossing) < 0.3:
                    intent = 0.9 if belief["answer"] == "cross" else 0.1
            crosses = bool(self.rng.random() < intent)
            direction = velocity / max(np.linalg.norm(velocity), 1e-6)
            destination = (
                position + direction * 5.0
                if np.linalg.norm(velocity) > 0.25
                else np.array([crossing[0], -np.sign(position[1] or 1.0) * 2.2])
            )
            if not crosses:
                destination = position + np.array([5.0, 0.0])
            actors.append(
                {
                    "position": position,
                    "velocity": velocity,
                    "destination": destination,
                    "kind": track["kind"],
                    "radius": 0.26 if track["kind"] == "worker" else 0.36,
                    "crosses": crosses,
                    "speed": max(0.2, float(np.linalg.norm(velocity))),
                }
            )
        occupancy = np.clip(
            c.hidden_occupancy_prior
            + c.hidden_uncertainty_weight * observation.get("blind_zone_uncertainty", 0.0),
            0,
            1,
        )
        for crossing in observation.get("crossings", []):
            if any(np.linalg.norm(actor["position"] - crossing) < 3.0 for actor in actors):
                continue
            if self.rng.random() < occupancy:
                side = self.rng.choice([-1.0, 1.0])
                position = np.asarray(crossing) + np.array(
                    [self.rng.normal(0, 0.15), side * self.rng.uniform(1.2, 2.5)]
                )
                crosses = bool(self.rng.random() < c.crossing_intent_prior)
                destination = (
                    np.asarray(crossing) + np.array([0.0, -side * 2.2])
                    if crosses
                    else position + np.array([5.0, 0.0])
                )
                actors.append(
                    {
                        "position": position,
                        "velocity": np.zeros(2),
                        "destination": destination,
                        "kind": "worker",
                        "radius": 0.26,
                        "crosses": crosses,
                        "speed": float(np.clip(self.rng.normal(0.65, 0.15), 0.25, 1.3)),
                    }
                )
        return actors

    def select(self, history, metadata, observation=None) -> Decision:
        observation = _observation_required(observation)
        c = self.config
        options = [("none", bridge, c.sim.commitment, 1.0) for bridge in ("cautious", "hold")]
        if observation.get("remaining_queries", 0) > 0:
            for source, information in metadata.items():
                if source not in ("infrastructure", "supervisor"):
                    continue
                if _available(information):
                    delays, weights = _latencies(information, c.latency_support)
                    options.extend(
                        (source, bridge, float(delay), float(weight))
                        for bridge in ("cautious", "hold")
                        for delay, weight in zip(delays, weights)
                    )
        outcomes = np.zeros((c.monte_carlo_samples, len(options), 3, 4))
        for replication in range(c.monte_carlo_samples):
            actors = self._sample_actors(observation)
            ticks = max(1, int(np.ceil(observation["remaining_time"] / c.sim.dt)))
            noise = self.rng.normal(size=(ticks, max(8, len(actors) + 4)))
            random = {
                source: self.rng.random(4)
                for source in metadata
                if source in ("infrastructure", "supervisor")
            }
            initial = _BeliefRollout(observation, actors, c.sim, noise)
            nominal = deepcopy(initial).run("nominal", 0.0, None)
            controls = {}
            for index, (source, bridge, delay, _) in enumerate(options):
                packet = (
                    None
                    if source == "none"
                    else initial.packet(source, metadata, delay, random[source])
                )
                duration = delay if source == "none" else min(delay, c.sim.request_expiry)
                # Match the learned scheduler's failure/no-response branch: an
                # absent reply cannot alter evidence, and its bridge lasts only
                # to the declared expiration before nominal continuation. The
                # runtime still re-evaluates every finite commitment interval.
                if packet is not None and not np.isfinite(packet.arrival_time):
                    duration = c.sim.request_expiry
                if packet is not None and packet.arrival_time > packet.expires_at:
                    packet = None
                key = (bridge, duration)
                if key not in controls:
                    controls[key] = deepcopy(initial).run(bridge, duration, None)
                query = (
                    controls[key]
                    if packet is None
                    else deepcopy(initial).run(bridge, duration, packet)
                )
                outcomes[replication, index] = np.stack((nominal, controls[key], query))
        scores = []
        for source, bridge in sorted({(source, bridge) for source, bridge, _, _ in options}):
            ids = [i for i, (s, b, _, _) in enumerate(options) if s == source and b == bridge]
            mass = np.asarray([options[i][3] for i in ids])
            mass /= mass.sum()
            info_draws = (outcomes[:, ids, 1, 0] - outcomes[:, ids, 2, 0]) @ mass
            motion_draws = (outcomes[:, ids, 0, 0] - outcomes[:, ids, 1, 0]) @ mass
            net_draws = info_draws + motion_draws - (c.query_cost if source != "none" else 0.0)
            risk_draws = outcomes[:, ids, 2, 1] @ mass
            value = float(
                net_draws.mean()
                - c.conservative_z * net_draws.std(ddof=1) / np.sqrt(len(net_draws))
            )
            risk = float(
                np.clip(
                    risk_draws.mean()
                    + c.conservative_z * risk_draws.std(ddof=1) / np.sqrt(len(risk_draws)),
                    0,
                    1,
                )
            )
            scores.append(
                {
                    "source": source,
                    "bridge": bridge,
                    "delay": float(np.asarray([options[i][2] for i in ids]) @ mass),
                    "net_value": value,
                    "collision_upper": risk,
                    "info": float(info_draws.mean()),
                    "motion": float(motion_draws.mean()),
                    "information_lower": float(
                        info_draws.mean()
                        - c.conservative_z * info_draws.std(ddof=1) / np.sqrt(len(info_draws))
                    ),
                    "completion": float((outcomes[:, ids, 2, 2] @ mass).mean()),
                    "time": float((outcomes[:, ids, 2, 3] @ mass).mean()),
                    "feasible": bool(
                        risk <= c.risk_limit
                        and (
                            source == "none"
                            or info_draws.mean()
                            - c.conservative_z * info_draws.std(ddof=1) / np.sqrt(len(info_draws))
                            > 0
                        )
                    ),
                }
            )
        feasible = [score for score in scores if score["feasible"]]
        controls = [score for score in scores if score["source"] == "none"]
        selected = (
            max(
                feasible,
                key=lambda score: (score["net_value"], score["completion"], -score["time"]),
            )
            if feasible
            else min(controls, key=lambda score: (score["collision_upper"], -score["net_value"]))
        )
        return Decision(
            selected["source"],
            selected["bridge"],
            selected["delay"],
            selected["net_value"],
            selected["collision_upper"],
            selected["feasible"],
            scores,
        )

    def reassess_packet(
        self,
        history,
        packet_metadata,
        now,
        source_metadata,
        latest_capture_time=None,
        observation=None,
    ):
        """Recompute expected information benefit with current onboard evidence.

        Packet defects remain unknown. The header age and all common freshness
        rules are observable; the subsequent common adapter fuses the payload.
        This gate uses a newly sampled current belief, not a cached initial score.
        """
        capture = float(packet_metadata["capture_time"])
        age = float(now) - capture
        if (
            age < 0
            or age > self.config.sim.max_reply_age
            or (latest_capture_time is not None and capture <= latest_capture_time)
        ):
            return {"accept": False, "reason": "stale_or_superseded"}
        observation = _observation_required(observation)
        source = packet_metadata["source"]
        metadata = {source: deepcopy(source_metadata[source])}
        metadata[source].update(latency_samples=[0.0], latencies=[0.0], capture_age=age)
        current = deepcopy(observation)
        # The already-acquired reply consumes no additional query budget.
        current["remaining_queries"] = max(1, current.get("remaining_queries", 0))
        result = self.select(history, metadata, current)
        scores = [
            score
            for score in result.candidate_scores
            if score["source"] == source
            and score["bridge"] == packet_metadata.get("bridge", "cautious")
        ]
        accepts = bool(
            scores
            and scores[0]["information_lower"] > 0
            and scores[0]["collision_upper"] <= self.config.risk_limit
        )
        return {
            "accept": accepts,
            "reason": "current_belief_useful" if accepts else "current_belief_not_useful",
            "capture_age": age,
        }


class DiagnosticOracle:
    """Privileged branch-cost selector; impossible to call as an ordinary policy.

    A runner supplies separately labelled counterfactual outcomes for the current
    state. No privileged record is packed into an observation/history. This is a
    headroom diagnostic, never a deployable or fair main-comparison baseline.
    """

    name = "diagnostic_oracle"

    def select_counterfactuals(
        self, records: list[dict], query_cost=0.03, risk_limit=0.1
    ) -> Decision:
        if not records:
            raise ValueError("Oracle requires explicitly privileged continuation labels")
        grouped = {}
        for record in records:
            key = (record["source"], record["bridge"], float(record["delay"]))
            grouped.setdefault(key, []).append(record)
        nominal_risk = float(np.mean([row["outcomes"]["A"]["collision"] for row in records]))
        scores = [
            {
                "source": "none",
                "bridge": "nominal",
                "delay": 0.0,
                "net_value": 0.0,
                "collision_upper": nominal_risk,
                "feasible": nominal_risk <= risk_limit,
            }
        ]
        for (source, bridge, delay), rows in grouped.items():
            score = {
                "source": source,
                "bridge": bridge,
                "delay": delay,
                "net_value": float(
                    np.mean(
                        [
                            row["outcomes"]["A"]["cost"] - row["outcomes"]["Q"]["cost"]
                            for row in rows
                        ]
                    )
                )
                - query_cost * (source != "none"),
                "collision_upper": float(
                    np.mean([row["outcomes"]["Q"]["collision"] for row in rows])
                ),
            }
            score["feasible"] = score["collision_upper"] <= risk_limit
            scores.append(score)
            scores.append(
                {
                    "source": "none",
                    "bridge": bridge,
                    "delay": delay,
                    "net_value": float(
                        np.mean(
                            [
                                row["outcomes"]["A"]["cost"] - row["outcomes"]["W"]["cost"]
                                for row in rows
                            ]
                        )
                    ),
                    "collision_upper": float(
                        np.mean([row["outcomes"]["W"]["collision"] for row in rows])
                    ),
                    "feasible": float(np.mean([row["outcomes"]["W"]["collision"] for row in rows]))
                    <= risk_limit,
                }
            )
        feasible = [score for score in scores if score["feasible"]]
        chosen = (
            max(feasible, key=lambda s: s["net_value"])
            if feasible
            else min(scores, key=lambda s: s["collision_upper"])
        )
        return Decision(**chosen, candidate_scores=scores)


MAIN_BASELINES = (
    "no_query_nominal",
    "no_query_cautious",
    "always_query",
    "confidence_triggered",
    "failure_lead_time",
    "planning_aware",
    "model_based_voi",
)


def make_baseline(
    name: str,
    config: dict | BaselineConfig | None = None,
    seed=0,
    *,
    failure_predictor=None,
    failure_predictor_provenance=None,
):
    name = {
        "nominal": "no_query_nominal",
        "cautious": "no_query_cautious",
        "confidence": "confidence_triggered",
    }.get(name, name)
    config = config if isinstance(config, BaselineConfig) else BaselineConfig.from_dict(config)
    if name == "model_based_voi":
        return ModelBasedVOI(config, seed)
    if name == "failure_lead_time":
        if failure_predictor is None:
            return RuleBaseline("failure_monitor_score", config)
        return FailureLeadTimeBaseline(failure_predictor, config, failure_predictor_provenance)
    if name == "failure_monitor_score":
        return RuleBaseline(name, config)
    if name not in MAIN_BASELINES:
        raise ValueError(f"Unknown deployable baseline {name}")
    return RuleBaseline(name, config)


def calibrate_source_metadata(
    development_verifications: list[dict], *, prior_alpha=1.0, prior_beta=1.0
) -> dict:
    """Freeze deployment-visible source quality estimates from independent audits.

    Each row records ``split='development'``, source, whether a response arrived,
    and, for informative responses, a Boolean correctness verification. Unknown
    answers have ``correct=None`` and are counted in response rate but not as
    fabricated correctness labels. Verifications may use offline simulator truth
    for development labels; no test-episode reliability is exposed to policies.
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior must be positive")
    if any(row.get("split") != "development" for row in development_verifications):
        raise ValueError("Source reliability calibration requires development-only audits")
    result = {}
    for source in ("infrastructure", "supervisor"):
        rows = [row for row in development_verifications if row["source"] == source]
        correctness = [
            bool(row["correct"])
            for row in rows
            if row.get("response", True) and row.get("correct") is not None
        ]
        alpha = prior_alpha + sum(correctness)
        beta_parameter = prior_beta + len(correctness) - sum(correctness)
        result[source] = {
            "reliability_prior": float(alpha / (alpha + beta_parameter)),
            "reliability_interval": beta.ppf([0.025, 0.975], alpha, beta_parameter).tolist(),
            "response_fraction": (1 + sum(bool(row.get("response", True)) for row in rows))
            / (2 + len(rows)),
            "calibration_verifications": len(correctness),
            "calibration_requests": len(rows),
            "calibration_split": "development",
            "calibration_status": "fitted" if correctness else "declared prior only",
        }
    return result
