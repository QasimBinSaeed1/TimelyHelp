"""A shared observation-only planner, evidence adapter, and braking safety layer."""

from __future__ import annotations
import math
import numpy as np
from .types import Control


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class LocalPlanner:
    def __init__(self, config):
        self.config = config
        self.last_control = Control()
        self.last_risk = 0.0

    def plan(self, observation, bridge="nominal"):
        c = self.config
        p = np.asarray(observation["pose"])
        g = np.asarray(observation["goal"])
        angle = math.atan2(g[1] - p[1], g[0] - p[0])
        heading = wrap(angle - p[2])
        speed = c.max_speed if bridge == "nominal" else c.cautious_speed
        if bridge == "hold":
            speed = 0.0
        speed *= max(0.0, math.cos(heading))
        risk = 0.0
        direction = np.array([math.cos(p[2]), math.sin(p[2])])
        v = observation.get("velocity", 0.0)
        for t in observation.get("tracks", []):
            position = np.asarray(t["position"])
            velocity = np.asarray(t["velocity"])
            age = max(0.0, observation["time"] - t["capture_time"])
            position = position + velocity * age
            rel = position - p[:2]
            along = float(rel @ direction)
            if -0.3 < along < 3.5:
                predicted = rel + velocity * max(0.0, along / max(v, 0.3))
                lateral = abs(float(predicted @ np.array([-direction[1], direction[0]])))
                margin = c.robot_radius + (0.26 if t["kind"] == "worker" else 0.36)
                margin += min(0.35, math.sqrt(t["covariance"]) + 0.07 * age)
                conflict = lateral < margin and along < 2.8
                if conflict:
                    risk = max(risk, max(0.0, 1.0 - along / 3.5))
                    speed = min(speed, max(0.0, (along - margin - 0.35) * 0.5))
        for belief in observation.get("intent_beliefs", []):
            if belief["answer"] != "cross":
                continue
            rel = np.asarray(belief["crossing"]) - p[:2]
            along = float(rel @ direction)
            if 0.0 < along < 2.8:
                speed = min(speed, max(0.0, (along - 1.1) * 0.45))
                risk = max(risk, 0.7)
        depth = observation.get("depth_min", c.sensor_range)
        # Sensor-based safety checks apply identically to all schedulers. Missing depth
        # expands uncertainty, rather than fabricating ground-truth obstacle distance.
        stopping = v * v / (2 * max(c.braking, 1e-3)) + c.robot_radius + 0.18
        if depth < stopping:
            speed = 0.0
            risk = max(risk, 0.9)
        if observation.get("dropout", False):
            speed = min(speed, c.cautious_speed)
        self.last_risk = risk
        self.last_control = Control(float(speed), float(np.clip(2.0 * heading, -1.2, 1.2)))
        return self.last_control


class EvidenceAdapter:
    """One freshness/association path shared by every baseline and learned gate."""

    def __init__(self, config):
        self.config = config
        self.intent_beliefs = []
        self.last_capture = {}

    def admit(self, message, tracker, now):
        if message.cancelled:
            return False, "cancelled"
        if now > message.expires_at + 1e-9:
            return False, "expired"
        if now - message.capture_time > self.config.max_reply_age + 1e-9:
            return False, "stale"
        if message.capture_time < self.last_capture.get(message.source, -float("inf")):
            return False, "newer_source_belief"
        p = message.payload
        if p.get("answer") == "unknown":
            return False, "unknown"
        if message.source == "infrastructure":
            detections = [
                {**d, "position": np.asarray(d["position"], float)} for d in p.get("tracks", [])
            ]
            # The tracker rejects captures older than an already-associated onboard observation.
            tracker.update(detections, now, provenance="infrastructure")
        elif message.source == "supervisor":
            crossing = p.get("crossing")
            if crossing is None:
                return False, "unknown"
            matching = [
                b
                for b in self.intent_beliefs
                if np.linalg.norm(np.array(b["crossing"]) - crossing) < 0.2
            ]
            if matching and matching[0]["capture_time"] > message.capture_time:
                return False, "newer_belief"
            self.intent_beliefs = [b for b in self.intent_beliefs if b not in matching]
            self.intent_beliefs.append(
                {
                    "crossing": crossing,
                    "answer": p["answer"],
                    "capture_time": message.capture_time,
                    "arrival_time": now,
                }
            )
        self.last_capture[message.source] = message.capture_time
        return True, "admitted"

    def current_intents(self, now):
        self.intent_beliefs = [
            b for b in self.intent_beliefs if now - b["capture_time"] <= self.config.max_reply_age
        ]
        return self.intent_beliefs


class ReactivePlanner(LocalPlanner):
    """Second frozen controller: depth steering and a conservative uncertainty speed.

    The inherited collision-braking checks and the evidence admission path remain
    identical. This is a declared second controller, not an adapted TIC-VLA claim.
    """

    def plan(self, observation, bridge="nominal"):
        action = super().plan(observation, bridge)
        if bridge != "hold":
            action.linear *= max(0.45, 1.0 - 0.45 * observation.get("blind_zone_uncertainty", 0.0))
            if observation.get("depth_min", 7.0) < 1.8:
                left = observation.get("depth_left", observation["depth_min"])
                right = observation.get("depth_right", observation["depth_min"])
                action.angular = float(
                    np.clip(action.angular + 0.5 * np.sign(left - right), -1.2, 1.2)
                )
        self.last_control = action
        return action


def make_planner(config):
    if config.controller == "local":
        return LocalPlanner(config)
    if config.controller == "reactive":
        return ReactivePlanner(config)
    raise ValueError(f"Unknown controller {config.controller}")
