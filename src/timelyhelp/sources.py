"""Finite questions, sensor-derived infrastructure replies, and bounded intent answers."""

from __future__ import annotations
import json
import numpy as np
from .types import Message

QUESTIONS = {
    "infrastructure": "local_tracks_at_crossing",
    "supervisor": "will_you_cross_delivery_lane",
}


class SourceModel:
    def __init__(self, config):
        self.config = config

    def sample_latency(self, env, source):
        samples = env.advertised_latency[source]
        index = min(len(samples) - 1, int(env.draw("latency", source) * len(samples)))
        delay = float(samples[index])
        disturbance = (
            "infrastructure_latency" if source == "infrastructure" else "supervisor_latency"
        )
        if disturbance in env.scenario.disturbances:
            delay = delay * 2.0 + 1.0 * env.draw("jitter", source)
        return delay

    def create(self, env, candidate, request_id, exact_delay=None):
        source = candidate.source
        if source not in QUESTIONS:
            raise ValueError("No-query candidates do not create messages")
        delay = self.sample_latency(env, source) if exact_delay is None else float(exact_delay)
        capture_time = env.time
        if source == "infrastructure":
            frames = env.source_frames
            lag = 1.8 if "stale_capture" in env.scenario.disturbances else 0.0
            eligible = [f for f in frames if f.capture_time <= env.time - lag + 1e-9]
            frame = eligible[-1] if eligible else frames[0]
            capture_time = frame.capture_time
            source_state = next(
                (
                    s
                    for s in reversed(env.source_track_history)
                    if abs(s["capture_time"] - capture_time) < 1e-9
                ),
                None,
            )
            tracks = (
                []
                if source_state is None
                else [
                    {**t, "position": list(t["position"]), "velocity": list(t["velocity"])}
                    for t in source_state["tracks"]
                ]
            )
            payload = {"tracks": tracks, "answer": "tracks" if tracks else "unknown"}
        else:
            # A worker knows only their own present destination. Asking never changes
            # that worker's policy, destination or random stream.
            worker = env.actors[0]
            permitted = (
                worker["kind"] == "worker"
                and np.linalg.norm(worker["position"] - env.geometry["crossings"][0]) < 4.5
            )
            answer = ("cross" if worker["crosses"] else "not_cross") if permitted else "unknown"
            if (
                "inaccurate_answer" in env.scenario.disturbances
                and env.draw("wrong_answer", request_id) < 0.55
            ):
                answer = {"cross": "not_cross", "not_cross": "cross", "unknown": "unknown"}[answer]
            elif env.draw("answer_error", request_id) < env.hidden_source_error["supervisor"]:
                answer = {"cross": "not_cross", "not_cross": "cross", "unknown": "unknown"}[answer]
            payload = {"answer": answer, "crossing": list(env.geometry["crossings"][0])}
        # Infrastructure failures affect what the detector sees / its calibrated frame,
        # and cannot expose a simulator corruption label to the receiver.
        if (
            source == "infrastructure"
            and env.draw("source_error", request_id) < env.hidden_source_error[source]
        ):
            for d in payload.get("tracks", []):
                d["position"][1] += 1.4 * (2 * env.draw("source_shift", request_id) - 1)
        lost = (
            "packet_loss" in env.scenario.disturbances
            and env.draw("packet_loss", request_id) < 0.45
        )
        no_response = (
            source == "supervisor"
            and "no_response" in env.scenario.disturbances
            and env.draw("no_response", request_id) < 0.6
        )
        arrival = float("inf") if lost or no_response else env.time + delay
        payload_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf8"))
        message = Message(
            request_id,
            source,
            QUESTIONS[source],
            float(capture_time),
            arrival,
            env.time + self.config.request_expiry,
            payload,
            payload_bytes,
        )
        return message
