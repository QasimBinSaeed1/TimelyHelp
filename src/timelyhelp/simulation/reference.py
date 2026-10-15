"""Deterministic observation-only CPU backend for correctness and pipeline tests.

This explicitly simplified 2-D backend is not Isaac Sim, DynaNav or Gazebo. It
provides exact branch restoration and reactive actors; native backends retain the
same observation and source interfaces and require separate replay audits.
"""

from __future__ import annotations
import copy
import hashlib
import math
import json
from dataclasses import asdict
import numpy as np
from ..types import SimConfig, Control, Candidate
from ..perception import ColorDepthDetector, ObservationTracker, render_rgbd
from ..control import EvidenceAdapter, wrap, make_planner
from ..sources import SourceModel
from .scenarios import Scenario, make_scenario

FEATURE_NAMES = (
    (
        "time_fraction",
        "odom_x",
        "odom_y",
        "heading_sin",
        "heading_cos",
        "speed",
        "last_angular",
        "goal_dx",
        "goal_dy",
        "front_depth",
        "track_count",
        "nearest_track",
        "track_uncertainty",
        "crossing_distance",
        "blind_zone_uncertainty",
        "ego_dropout",
        "infrastructure_coverage",
        "supervisor_coverage",
        "infrastructure_delay_mean",
        "supervisor_delay_mean",
        "infrastructure_delay_std",
        "supervisor_delay_std",
        "infrastructure_response_fraction",
        "supervisor_response_fraction",
        "infrastructure_capture_age",
        "supervisor_capture_age",
        "infrastructure_pending",
        "supervisor_pending",
        "remaining_time_fraction",
        "remaining_query_fraction",
        "infrastructure_reliability_prior",
        "supervisor_reliability_prior",
    )
    + tuple(
        f"track_{rank}_{component}"
        for rank in range(1, 4)
        for component in (
            "along",
            "lateral",
            "forward_velocity",
            "lateral_velocity",
            "covariance",
            "capture_age",
            "kind",
            "source",
        )
    )
    + (
        "admitted_intent_cross",
        "admitted_intent_clear",
        "admitted_intent_age",
        "left_depth",
        "right_depth",
    )
)


class ReferenceSimulator:
    backend_name = "reference"

    def __init__(self, scenario=None, seed=0, config=None):
        self.scenario = (
            Scenario.from_dict(scenario)
            if isinstance(scenario, dict)
            else (scenario or make_scenario())
        )
        self.config = (
            SimConfig.from_dict(config) if isinstance(config, dict) else (config or SimConfig())
        )
        self.seed = int(seed)
        self.replication = 0
        self.tick = 0
        self.time = 0.0
        self.geometry = self.scenario.geometry()
        self.pose = np.asarray(self.geometry["start"], float)
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.goal = np.asarray(self.geometry["goal"], float)
        self.last_control = Control()
        self.collision = False
        self.completion = False
        self.done = False
        self.distance = 0.0
        self.near_misses = 0
        self._near_contact = False
        self.queries = 0
        self.communication_bytes = 0
        self.workload = 0
        self.reply_delays = []
        self.pending = []
        self.received = []
        self._request_id = 0
        self.source_frames = []
        self.source_track_history = []
        self.frames = []
        self.history = []
        self.control_log = []
        self.request_log = []
        self.bridge_log = []
        self._next_bridge = None
        self.admission_filter = None
        # Independently randomize speed assignment and source view availability.
        swapped = self.draw("episode_latency_assignment", tick=0) < 0.5
        self.advertised_latency = {
            source: [
                float(x) + self.config.fusion_time
                for x in self.config.source_latency[
                    ("supervisor" if source == "infrastructure" else "infrastructure")
                    if swapped
                    else source
                ]
            ]
            for source in ("infrastructure", "supervisor")
        }
        if self.draw("episode_source_view", tick=0) < 0.25:
            self.geometry["camera_pose"][2] += np.pi
        self.detector = ColorDepthDetector(self.config.camera_fov, self.config.detector_noise)
        self.tracker = ObservationTracker(self.config.track_timeout)
        self.source_tracker = ObservationTracker(self.config.track_timeout)
        self.planner = make_planner(self.config)
        self.evidence = EvidenceAdapter(self.config)
        self.source_model = SourceModel(self.config)
        self.source_stats = {
            s: {
                "latencies": list(self.advertised_latency[s]),
                "requests": 0,
                "responses": 0,
                "last_capture": None,
                "last_arrival": None,
                "status": "available",
                "reliability_prior": 0.8,
            }
            for s in ("infrastructure", "supervisor")
        }
        # Reliability is a hidden episode parameter independent of public source identity/latency.
        self.hidden_source_error = {
            s: 0.02 + 0.23 * self.draw("episode_source_error", s, tick=0)
            for s in ("infrastructure", "supervisor")
        }
        self.actors = self._create_actors()
        self._sense()
        self._record_history()

    def draw(self, channel, index=0, tick=None):
        """Counter-based common randomness: branch actions never advance draw streams."""
        tick = self.tick if tick is None else tick
        data = f"{self.seed}:{self.replication}:{tick}:{channel}:{index}".encode()
        integer = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")
        return (integer + 0.5) / 18446744073709551616.0

    def normal(self, channel, index=0, tick=None):
        a = max(self.draw(channel, str(index) + "a", tick), 1e-15)
        b = self.draw(channel, str(index) + "b", tick)
        return math.sqrt(-2 * math.log(a)) * math.cos(2 * math.pi * b)

    def _create_actors(self):
        side = self.geometry["side"]
        crossing = self.geometry["crossing"]
        family = self.scenario.family
        crosses = family != "ambiguous_intent" or self.draw("intended_destination", tick=0) > 0.5
        kind = "cart" if family == "cart_junction" else "worker"
        pos = np.array([crossing, side * (1.35 if family == "ambiguous_intent" else 2.2)])
        destination = (
            np.array([crossing, -side * 2.2])
            if crosses
            else np.array([crossing + 4.5, side * 1.35])
        )
        actors = [
            {
                "position": pos,
                "velocity": np.zeros(2),
                "destination": destination,
                "kind": kind,
                "radius": 0.36 if kind == "cart" else 0.26,
                "crosses": crosses,
                "speed": (0.8 if kind == "cart" else 0.65)
                * (0.8 + 0.4 * self.draw("actor_speed", tick=0)),
                "phase": self.draw("actor_phase", tick=0),
            }
        ]
        if "crowd" in self.scenario.disturbances:
            for i in range(3):
                x = 7.0 + i * 0.8
                actors.append(
                    {
                        "position": np.array([x, -side * 1.8]),
                        "velocity": np.zeros(2),
                        "destination": np.array([x, side * 2.2]),
                        "kind": "worker",
                        "radius": 0.26,
                        "crosses": True,
                        "speed": 0.5 + 0.25 * self.draw("crowd_speed", i, tick=0),
                        "phase": self.draw("crowd_phase", i, tick=0),
                    }
                )
        return actors

    def _render(self, pose, source="onboard"):
        rectangles = list(self.geometry["rectangles"])
        if source == "infrastructure" and "source_occlusion" in self.scenario.disturbances:
            x = self.geometry["crossing"]
            side = self.geometry["side"]
            rectangles.append(
                (x - 0.6, min(-side * 1.2, -side * 0.6), x + 0.6, max(-side * 1.2, -side * 0.6))
            )
        return render_rgbd(
            pose,
            rectangles,
            self.actors,
            self.time,
            self.config.camera_width,
            self.config.camera_height,
            self.config.camera_fov,
            self.config.sensor_range,
        )

    def _sense(self):
        frame = self._render(self.pose, "onboard")
        if "calibration_error" in self.scenario.disturbances:
            old_pose = frame.pose.copy()
            frame.pose = frame.pose + np.array([0.0, 0.15, 0.055])
            if frame.xy_map is not None:
                rotation = np.array(
                    [[math.cos(0.055), -math.sin(0.055)], [math.sin(0.055), math.cos(0.055)]]
                )
                frame.xy_map = (frame.xy_map - old_pose[:2]) @ rotation.T + frame.pose[:2]
        if "dropout" in self.scenario.disturbances and self.draw("ego_dropout") < 0.4:
            frame.dropped = True
        self.frames.append(frame)
        self.frames = self.frames[-80:]
        self.frame = frame
        self.tracker.update(
            self.detector.detect(frame, lambda i: self.normal("ego_detector", i)), self.time
        )
        infra = self._render(np.asarray(self.geometry["camera_pose"], float), "infrastructure")
        self.source_frames.append(infra)
        self.source_frames = self.source_frames[-80:]
        self.source_tracker.update(
            self.detector.detect(infra, lambda i: self.normal("source_detector", i)),
            self.time,
            provenance="infrastructure",
        )
        self.source_track_history.append(
            {
                "capture_time": infra.capture_time,
                "tracks": [t.public() for t in self.source_tracker.tracks],
            }
        )
        self.source_track_history = self.source_track_history[-80:]

    def _physics_step(self, control):
        """Extension hook: reference integrates differential-drive and reactive actors."""
        c = self.config
        dt = c.dt
        payload = "payload_dynamics" in self.scenario.disturbances
        acceleration = c.acceleration * (0.45 if payload else 1.0)
        braking = c.braking * (0.4 if payload else 1.0)
        target = float(np.clip(control.linear, 0, c.max_speed))
        limit = acceleration if target > self.velocity else braking
        self.velocity += float(np.clip(target - self.velocity, -limit * dt, limit * dt))
        self.angular_velocity = float(np.clip(control.angular, -1.2, 1.2))
        self.pose[2] = wrap(self.pose[2] + self.angular_velocity * dt)
        displacement = (
            self.velocity * dt * np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
        )
        self.pose[:2] += displacement
        self.distance += float(np.linalg.norm(displacement))
        for i, actor in enumerate(self.actors):
            delta = actor["destination"] - actor["position"]
            dist = np.linalg.norm(delta)
            desired = actor["speed"] * delta / max(dist, 1e-9)
            # Actor reacts to each branch's real robot state; noise alone is held common.
            robot_rel = actor["position"] - self.pose[:2]
            robot_distance = np.linalg.norm(robot_rel)
            if robot_distance < 1.6:
                away = robot_rel / max(robot_distance, 1e-6)
                desired += 0.25 * (1.6 - robot_distance) * away
                desired *= 0.75 if self.velocity > 0.1 else 1.0
            jitter = 0.045 * self.normal("actor_heading", i)
            desired += jitter * np.array([-delta[1], delta[0]]) / max(dist, 1e-9)
            actor["velocity"] = desired if dist > 0.15 else np.zeros(2)
            actor["position"] += actor["velocity"] * dt
        self.tick += 1
        self.time = self.tick * dt
        gaps = [
            np.linalg.norm(a["position"] - self.pose[:2]) - a["radius"] - c.robot_radius
            for a in self.actors
        ]
        for rectangle in self.geometry["rectangles"]:
            closest = np.clip(self.pose[:2], rectangle[:2], rectangle[2:])
            gaps.append(np.linalg.norm(self.pose[:2] - closest) - c.robot_radius)
        gap = min(gaps, default=10.0)
        contact = gap < c.near_miss_distance
        if contact and not self._near_contact:
            self.near_misses += 1
        self._near_contact = contact
        if gap <= 0:
            self.collision = True
        if not self.collision and np.linalg.norm(self.pose[:2] - self.goal) < 0.35:
            self.completion = True
        self.done = self.collision or self.completion or self.time >= c.horizon - 1e-9

    def step(self, control):
        if self.done:
            return self.observe()
        if not isinstance(control, Control):
            control = Control(**control)
        self.last_control = copy.deepcopy(control)
        self.control_log.append(asdict(control))
        self.bridge_log.append(self._next_bridge)
        self._next_bridge = None
        self._physics_step(control)
        self._sense()
        self._process_messages()
        self._record_history()
        return self.observe()

    def advance(self, bridge="nominal"):
        if bridge not in ("nominal", "cautious", "hold"):
            raise ValueError("Unknown bridge")
        self._process_messages()
        self._next_bridge = bridge
        return self.step(self.planner.plan(self.observe(), bridge))

    def request(self, candidate, exact_delay=None):
        if candidate.source == "none":
            return None
        if self.done or self.queries >= self.config.max_queries:
            return None
        for previous in self.pending:
            if previous.source == candidate.source and previous.reason == "pending":
                self.cancel(previous.request_id)
        self._request_id += 1
        self.queries += 1
        self.source_stats[candidate.source]["requests"] += 1
        self.communication_bytes += len(
            json.dumps(
                {
                    "source": candidate.source,
                    "question": candidate.source,
                    "request_id": self._request_id,
                },
                separators=(",", ":"),
            ).encode("utf8")
        )
        if candidate.source == "supervisor":
            self.workload += 1
        message = self.source_model.create(self, candidate, self._request_id, exact_delay)
        self.pending.append(message)
        self.request_log.append(
            {
                "tick": self.tick,
                "candidate": candidate.to_dict(),
                "exact_delay": exact_delay,
                "request_id": message.request_id,
                "action": "request",
            }
        )
        self._process_messages()
        return message.request_id

    def cancel(self, request_id):
        for message in self.pending:
            if message.request_id == request_id and message.reason == "pending":
                message.cancelled = True
                message.reason = "cancelled"
                self.request_log.append(
                    {"tick": self.tick, "request_id": request_id, "action": "cancel"}
                )
                return True
        return False

    def _process_messages(self):
        for message in self.pending:
            if message.reason != "pending":
                continue
            if self.time > message.expires_at + 1e-9:
                message.reason = "expired"
                continue
            if message.arrival_time > self.time + 1e-9:
                continue
            # Stamp actual admission tick separately from source capture and scheduled transport.
            arrival = self.time
            stats = self.source_stats[message.source]
            self.reply_delays.append(arrival - (message.expires_at - self.config.request_expiry))
            stats["latencies"].append(self.reply_delays[-1])
            stats["latencies"] = stats["latencies"][-32:]
            stats["responses"] += 1
            stats["last_capture"] = message.capture_time
            stats["last_arrival"] = arrival
            self.communication_bytes += message.bytes
            message.arrival_time = arrival
            if self.admission_filter is not None and not self.admission_filter(
                message.public(), self.observe()
            ):
                message.reason = "predicted_useless"
                message.admitted = False
            else:
                message.admitted, message.reason = self.evidence.admit(
                    message, self.tracker, self.time
                )
            self.received.append(message.public())

    def observe(self):
        c = self.config
        tracks = [t.public() for t in self.tracker.tracks]
        sources = {}
        for source, stats in self.source_stats.items():
            active = [m for m in self.pending if m.source == source and m.reason == "pending"]
            expired = sum(m.source == source and m.reason == "expired" for m in self.pending)
            resolved = stats["responses"] + expired
            alpha = stats.get("response_prior_alpha", 1.0)
            beta = stats.get("response_prior_beta", 0.0)
            camera = self.geometry["camera_pose"]
            crossing = np.asarray(self.geometry["crossings"][0])
            bearing = math.atan2(crossing[1] - camera[1], crossing[0] - camera[0])
            coverage = 0.75 if abs(wrap(bearing - camera[2])) < c.camera_fov / 2 else 0.05
            sources[source] = {
                "coverage": coverage if source == "infrastructure" else 0.35,
                "latencies": list(stats["latencies"]),
                "response_fraction": (alpha + stats["responses"]) / (alpha + beta + resolved),
                "status": "pending" if active else "available",
                "last_capture": stats["last_capture"],
                "last_arrival": stats["last_arrival"],
                "reliability_prior": stats["reliability_prior"],
                "reliability_calibration_count": stats.get("reliability_calibration_count", 0),
                "reliability_calibration": stats.get("reliability_calibration", "declared_prior"),
                "camera_pose": list(self.geometry["camera_pose"])
                if source == "infrastructure"
                else None,
                "fov": c.camera_fov if source == "infrastructure" else None,
                "range": c.sensor_range if source == "infrastructure" else None,
            }
        # Visibility uncertainty comes from sensed static material close to the image
        # corridor, plus image coverage of the public junction; it contains no actor truth.
        # The wide vertical field sees the vehicle body and immediate ground below the
        # horizon. Forward clearance uses rows at and above the optical center so that
        # self-return does not trip the shared braking check.
        height = self.frame.depth.shape[0]
        middle = self.frame.depth[
            int(height * 0.15) : max(int(height * 0.50), int(height * 0.15) + 1), :
        ]
        center = middle[:, int(c.camera_width * 0.42) : int(c.camera_width * 0.58)]
        depth_min = c.sensor_range if self.frame.dropped else float(np.min(center))
        gray = (np.max(self.frame.rgb, axis=2) - np.min(self.frame.rgb, axis=2)) < 12
        static_fraction = 0.0 if self.frame.dropped else float(np.mean(gray))
        crossing_dist = float(
            np.linalg.norm(np.asarray(self.geometry["crossings"][0]) - self.pose[:2])
        )
        uncertainty = float(
            np.clip(static_fraction + 0.3 * (crossing_dist > 3.0) + 0.5 * self.frame.dropped, 0, 1)
        )
        return {
            "time": float(self.time),
            "pose": self.pose.tolist(),
            "velocity": float(self.velocity),
            "angular_velocity": float(self.angular_velocity),
            "goal": self.goal.tolist(),
            "tracks": tracks,
            "depth_min": depth_min,
            "depth_left": c.sensor_range
            if self.frame.dropped
            else float(np.median(middle[:, : c.camera_width // 3])),
            "depth_right": c.sensor_range
            if self.frame.dropped
            else float(np.median(middle[:, -c.camera_width // 3 :])),
            "dropout": bool(self.frame.dropped),
            "sources": sources,
            "crossings": copy.deepcopy(self.geometry["crossings"]),
            "map_obstacles": copy.deepcopy(self.geometry["rectangles"]),
            "blind_zone_uncertainty": uncertainty,
            "intent_beliefs": copy.deepcopy(self.evidence.current_intents(self.time)),
            "remaining_time": max(0.0, c.horizon - self.time),
            "remaining_queries": max(0, c.max_queries - self.queries),
            "pending": [
                {
                    "request_id": m.request_id,
                    "source": m.source,
                    "reason": m.reason,
                    "cancelled": m.cancelled,
                    "expires_at": m.expires_at,
                }
                for m in self.pending
            ],
            "messages": copy.deepcopy(self.received[-8:]),
        }

    def _feature(self):
        o = self.observe()
        c = self.config
        p = o["pose"]
        s = o["sources"]
        tracks = o["tracks"]
        distances = [np.linalg.norm(np.asarray(t["position"]) - p[:2]) for t in tracks]
        ages = [
            c.max_reply_age if s[k]["last_capture"] is None else self.time - s[k]["last_capture"]
            for k in ("infrastructure", "supervisor")
        ]
        values = [
            self.time / c.horizon,
            p[0] / 12,
            p[1] / 3,
            math.sin(p[2]),
            math.cos(p[2]),
            self.velocity,
            self.last_control.angular,
            (self.goal[0] - p[0]) / 12,
            (self.goal[1] - p[1]) / 3,
            o["depth_min"] / c.sensor_range,
            len(tracks) / 5,
            min(distances, default=c.sensor_range) / c.sensor_range,
            np.mean([t["covariance"] for t in tracks]) if tracks else 1.0,
            np.linalg.norm(np.array(o["crossings"][0]) - p[:2]) / 12,
            o["blind_zone_uncertainty"],
            float(o["dropout"]),
            s["infrastructure"]["coverage"],
            s["supervisor"]["coverage"],
        ]
        for statistic in (np.mean, np.std):
            values.extend(
                float(statistic(s[k]["latencies"])) / 6 for k in ("infrastructure", "supervisor")
            )
        values.extend(s[k]["response_fraction"] for k in ("infrastructure", "supervisor"))
        values.extend(age / 6 for age in ages)
        values.extend(float(s[k]["status"] == "pending") for k in ("infrastructure", "supervisor"))
        values.extend(
            [o["remaining_time"] / c.horizon, o["remaining_queries"] / max(1, c.max_queries)]
        )
        values.extend(s[k]["reliability_prior"] for k in ("infrastructure", "supervisor"))
        # Preserve decision-relevant received evidence in the frozen encoder.
        # Ordering uses observed geometry only; no simulator or tracker ID enters
        # the model. Public class/source codes distinguish padded empty slots.
        forward = np.array([math.cos(p[2]), math.sin(p[2])])
        lateral = np.array([-forward[1], forward[0]])
        nearest = sorted(
            tracks,
            key=lambda track: (
                float(np.linalg.norm(np.asarray(track["position"]) - p[:2])),
                tuple(track["position"]),
                track["kind"],
                track["provenance"],
            ),
        )[:3]
        for track in nearest:
            age = max(0.0, self.time - float(track["capture_time"]))
            velocity = np.asarray(track["velocity"], float)
            relative = np.asarray(track["position"], float) + velocity * age - p[:2]
            values.extend(
                [
                    float(relative @ forward) / c.sensor_range,
                    float(relative @ lateral) / c.sensor_range,
                    float(velocity @ forward),
                    float(velocity @ lateral),
                    float(track["covariance"]),
                    age / 6.0,
                    {"worker": 1.0, "cart": 2.0}.get(track["kind"], 0.0),
                    {"onboard": 1.0, "infrastructure": 2.0}.get(track["provenance"], 0.0),
                ]
            )
        values.extend([0.0] * (8 * (3 - len(nearest))))
        intents = o.get("intent_beliefs", [])
        intent = (
            min(
                intents,
                key=lambda belief: float(np.linalg.norm(np.asarray(belief["crossing"]) - p[:2])),
            )
            if intents
            else None
        )
        values.extend(
            [
                float(intent is not None and intent["answer"] == "cross"),
                float(intent is not None and intent["answer"] in ("not_cross", "clear")),
                max(0.0, self.time - float(intent["capture_time"])) / 6.0 if intent else 0.0,
                o["depth_left"] / c.sensor_range,
                o["depth_right"] / c.sensor_range,
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def _record_history(self):
        self.history.append(self._feature())
        self.history = self.history[-self.config.history_length :]
        self._last_history_tick = self.tick

    def current_history_features(self):
        """Current onboard evidence before admitting an asynchronous arrival."""
        history = list(self.history)
        if getattr(self, "_last_history_tick", -1) == self.tick:
            if history:
                history[-1] = self._feature()
            else:
                history = [self._feature()]
        else:
            history.append(self._feature())
        history = history[-self.config.history_length :]
        history = [history[0]] * (self.config.history_length - len(history)) + history
        return np.stack(history).astype(np.float32)

    def history_features(self):
        if not self.history:
            raise RuntimeError("No observations captured")
        history = [self.history[0]] * (
            self.config.history_length - len(self.history)
        ) + self.history
        return np.stack(history).astype(np.float32)

    def outcomes(self):
        c = self.config
        cost = (
            c.collision_weight * float(self.collision)
            + c.failure_weight * float(not self.completion)
            + c.time_weight * self.time / c.horizon
        )
        return {
            "completion": bool(self.completion),
            "collision": bool(self.collision),
            "time": float(self.time),
            "completion_time": float(self.time),
            "distance": float(self.distance),
            "near_misses": int(self.near_misses),
            "queries": int(self.queries),
            "reply_delays": list(self.reply_delays),
            "communication_bytes": int(self.communication_bytes),
            "workload": int(self.workload),
            "cost": float(cost),
        }

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot):
        self.__dict__ = copy.deepcopy(snapshot)
        return self

    def spawn_from_snapshot(self, snapshot):
        return self.fork(snapshot)

    def fork(self, snapshot=None):
        new = object.__new__(type(self))
        new.restore(snapshot or self.snapshot())
        return new

    def reset_replay(self, snapshot):
        """Rebuild common prefix including requests/cancels, sensor buffers and controller."""
        new = type(self)(snapshot["scenario"], snapshot["seed"], snapshot["config"])
        new.replication = snapshot["replication"]
        # Replications normally begin after the snapshot. A nonzero prefix requires
        # regenerating initial sensors and actors under that replication as well.
        if new.replication:
            new.actors = new._create_actors()
            new.frames = []
            new.source_frames = []
            new.history = []
            new.tracker = ObservationTracker(new.config.track_timeout)
            new.source_tracker = ObservationTracker(new.config.track_timeout)
            new.source_track_history = []
            new._sense()
            new._record_history()
        events = list(snapshot["request_log"])
        for tick, control in enumerate(snapshot["control_log"]):
            for event in events:
                if event["tick"] == tick:
                    if event["action"] == "request":
                        new.request(Candidate(**event["candidate"]), event["exact_delay"])
                    else:
                        new.cancel(event["request_id"])
            bridge = snapshot.get("bridge_log", [None] * len(snapshot["control_log"]))[tick]
            if bridge is None:
                new.step(Control(**control))
            else:
                new.advance(bridge)
        for event in events:
            if event["tick"] == len(snapshot["control_log"]):
                if event["action"] == "request":
                    new.request(Candidate(**event["candidate"]), event["exact_delay"])
                else:
                    new.cancel(event["request_id"])
        return new
