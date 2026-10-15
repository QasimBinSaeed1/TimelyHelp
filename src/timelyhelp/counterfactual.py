"""Controlled A/W/Q continuations with exact restoration and prearrival audits."""

from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from .types import Candidate


class CounterfactualRunner:
    def __init__(self, env):
        self.env = env
        self._baseline_cache = {}
        self._motion_cache = {}
        self._cached_snapshot = None

    def _spawn(self, snapshot):
        hook = getattr(self.env, "spawn_from_snapshot", None)
        return hook(snapshot) if hook is not None else self.env.fork(snapshot)

    def _tolerances(self):
        return getattr(
            self.env,
            "audit_tolerances",
            {
                "pose": 1e-12,
                "control": 1e-12,
                "rgb_max": 0.0,
                "rgb_changed_fraction": 0.0,
                "depth_max": 0.0,
                "depth_valid_fraction": 0.0,
            },
        )

    def _sensor_compare(self, a, b):
        if a["rgb"].shape != b["rgb"].shape or a["depth"].shape != b["depth"].shape:
            return False, {}
        tolerances = self._tolerances()
        rgb = np.abs(a["rgb"].astype(float) - b["rgb"].astype(float))
        finite = np.isfinite(a["depth"]) & np.isfinite(b["depth"])
        diff = np.abs(a["depth"][finite] - b["depth"][finite])
        errors = {
            "rgb_max": float(rgb.max(initial=0.0)),
            "rgb_changed_fraction": float(np.mean(np.any(rgb > 0, axis=-1))),
            "depth_max": float(diff.max(initial=0.0)),
            "depth_valid_fraction": float(
                np.mean(np.isfinite(a["depth"]) != np.isfinite(b["depth"]))
            ),
        }
        passed = (
            all(errors[key] <= tolerances[key] for key in errors) and a["dropped"] == b["dropped"]
        )
        return passed, errors

    def _sample(self, env):
        return {
            "time": env.time,
            "pose": env.pose.copy(),
            "control": copy.deepcopy(env.last_control),
            "rgb": env.frame.rgb.copy(),
            "depth": env.frame.depth.copy(),
            "dropped": env.frame.dropped,
        }

    def _identifier(self, snapshot):
        descriptor = {
            "scenario": snapshot["scenario"].to_dict(),
            "seed": snapshot["seed"],
            "tick": snapshot["tick"],
            "controls": snapshot["control_log"],
            "requests": snapshot["request_log"],
            "config": snapshot["config"].to_dict(),
            "backend": self.env.backend_name,
        }
        return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]

    def _continuation(self, snapshot, candidate, branch, replication):
        env = self._spawn(snapshot)
        try:
            env.replication = replication
            start = env.time
            prefix = self._sample(env)
            trace = []
            if branch == "Q" and candidate.source != "none":
                env.request(candidate, exact_delay=candidate.delay)
            while not env.done:
                elapsed = env.time - start
                bridge = (
                    candidate.bridge
                    if branch in ("W", "Q") and elapsed < candidate.delay - 1e-9
                    else "nominal"
                )
                # Zero-delay Q admits evidence before its first control.
                env.advance(bridge)
                trace.append(self._sample(env))
            return env.outcomes(), trace, env, prefix
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    def run(self, snapshot, candidate, replication=0):
        if not isinstance(candidate, Candidate):
            candidate = Candidate(**candidate)
        snapshot_id = self._identifier(snapshot)
        if snapshot_id != self._cached_snapshot:
            self._baseline_cache.clear()
            self._motion_cache.clear()
            self._cached_snapshot = snapshot_id
        base_key = (snapshot_id, int(replication))
        motion_key = base_key + (candidate.bridge, float(candidate.delay))
        if base_key not in self._baseline_cache:
            self._baseline_cache[base_key] = self._continuation(
                snapshot, candidate, "A", replication
            )
        if motion_key not in self._motion_cache:
            self._motion_cache[motion_key] = self._continuation(
                snapshot, candidate, "W", replication
            )
        a, _, _, _ = self._baseline_cache[base_key]
        w, wtrace, _, wprefix = self._motion_cache[motion_key]
        q, qtrace, qenv, qprefix = self._continuation(snapshot, candidate, "Q", replication)
        start = snapshot["time"]
        checked = 0
        max_pose_error = 0.0
        max_control_error = 0.0
        sensor_equal, sensor_errors = self._sensor_compare(wprefix, qprefix)
        prefix_equal = sensor_equal and np.allclose(
            wprefix["pose"], qprefix["pose"], atol=self._tolerances()["pose"], rtol=0
        )

        # A shorter trace must not pass merely because zip ignores its missing
        # samples. Only controls starting before admission belong to this audit.
        def before_admission(trace):
            return [
                sample
                for sample in trace
                if sample["time"] - snapshot["config"].dt < start + candidate.delay - 1e-9
            ]

        wbefore, qbefore = before_admission(wtrace), before_admission(qtrace)
        aligned = len(wbefore) == len(qbefore) and all(
            abs(wt["time"] - qt["time"]) <= 1e-9 for wt, qt in zip(wbefore, qbefore)
        )
        # Check controls whose execution began strictly before the first possible
        # admission. Sensors at the admission tick are also produced before fusion.
        for wt, qt in zip(wbefore, qbefore):
            checked += 1
            pose_error = float(np.max(np.abs(wt["pose"] - qt["pose"])))
            control_error = max(
                abs(wt["control"].linear - qt["control"].linear),
                abs(wt["control"].angular - qt["control"].angular),
            )
            max_pose_error = max(max_pose_error, pose_error)
            max_control_error = max(max_control_error, control_error)
            matched, errors = self._sensor_compare(wt, qt)
            for key, value in errors.items():
                sensor_errors[key] = max(sensor_errors[key], value)
            sensor_equal = sensor_equal and matched
        tolerances = self._tolerances()
        passed = (
            prefix_equal
            and aligned
            and max_pose_error <= tolerances["pose"]
            and max_control_error <= tolerances["control"]
            and sensor_equal
        )
        if not passed:
            raise AssertionError(
                "W/Q prearrival physical control or onboard sensor mismatch "
                f"ticks={checked} pose={max_pose_error:.6g} control={max_control_error:.6g} "
                f"prefix_equal={prefix_equal} aligned={aligned} "
                f"sensors_equal={sensor_equal} sensor_errors={sensor_errors} delay={float(candidate.delay):g} "
                f"source={candidate.source} bridge={candidate.bridge} rep={int(replication)}"
            )
        from .simulation.reference import ReferenceSimulator

        # Conditioning is the recorded common prefix, not a fresh native replay.
        # Replay deviations are audited separately and must not change MC groups.
        obs_env = object.__new__(ReferenceSimulator)
        obs_env.restore(snapshot)
        observation = obs_env.observe()
        rollout_descriptor = {
            "layout": snapshot["scenario"].layout_id,
            "seed": snapshot["seed"],
            "family": snapshot["scenario"].family,
            "disturbances": snapshot["scenario"].disturbances,
            "controller": snapshot["config"].controller,
            "backend": self.env.backend_name,
        }
        rollout_id = hashlib.sha256(
            json.dumps(rollout_descriptor, sort_keys=True).encode()
        ).hexdigest()[:24]
        messages = [m.public() for m in qenv.pending if m.request_id > snapshot["_request_id"]]
        # A no-response message has no arrival timestamp to expose in serialized data.
        for message in messages:
            if not np.isfinite(message["arrival_time"]):
                message["arrival_time"] = None
        history = obs_env.history_features().tolist()
        close = getattr(obs_env, "close", None)
        if close is not None:
            close()
        return {
            "history": history,
            "observation": observation,
            "source": candidate.source,
            "bridge": candidate.bridge,
            "delay": float(candidate.delay),
            "layout": snapshot["scenario"].layout_id,
            "setting": snapshot["scenario"].setting,
            "family": snapshot["scenario"].family,
            "disturbances": list(snapshot["scenario"].disturbances),
            "controller": snapshot["config"].controller,
            "rollout_id": rollout_id,
            "snapshot_id": snapshot_id,
            "split": snapshot["scenario"].split,
            "replication": int(replication),
            "seed": snapshot["seed"],
            "backend": self.env.backend_name,
            "outcomes": {"A": a, "W": w, "Q": q},
            "costs": {"A": a["cost"], "W": w["cost"], "Q": q["cost"]},
            "info": float(w["cost"] - q["cost"]),
            "motion": float(a["cost"] - w["cost"]),
            "net": float(
                a["cost"] - q["cost"] - snapshot["config"].query_cost * (candidate.source != "none")
            ),
            "messages": messages,
            "prearrival_audit": {
                "passed": passed,
                "ticks_checked": checked,
                "max_pose_error": max_pose_error,
                "max_control_error": max_control_error,
                "sensors_equal": sensor_equal,
                "prefix_equal": bool(prefix_equal),
                "traces_aligned": bool(aligned),
                "sensor_errors": sensor_errors,
                "tolerances": tolerances,
            },
        }

    def audit_restore(self, snapshot, steps=5):
        traces = []
        prefixes = []
        tolerances = self._tolerances()
        for method in ("snapshot", "snapshot", "replay"):
            env = self._spawn(snapshot) if method == "snapshot" else self.env.reset_replay(snapshot)
            prefixes.append(self._sample(env))
            trace = []
            for _ in range(steps):
                if env.done:
                    break
                env.advance("nominal")
                trace.append(self._sample(env))
            traces.append(trace)
            close = getattr(env, "close", None)
            if close is not None:
                close()
        max_error = 0.0
        max_control_error = 0.0
        sensors_equal = True
        prefix_equal = True
        observed_sensor_errors = {
            key: 0.0
            for key in ("rgb_max", "rgb_changed_fraction", "depth_max", "depth_valid_fraction")
        }
        for prefix in prefixes[1:]:
            match, errors = self._sensor_compare(prefixes[0], prefix)
            prefix_equal = (
                prefix_equal
                and match
                and np.allclose(
                    prefixes[0]["pose"], prefix["pose"], atol=tolerances["pose"], rtol=0
                )
            )
            for key, value in errors.items():
                observed_sensor_errors[key] = max(observed_sensor_errors[key], value)
            if not match and self.env.backend_name != "reference":
                from pathlib import Path

                debug = (
                    Path("artifacts/native") / f"{self.env.backend_name}_replay_rgb_mismatch.npz"
                )
                debug.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    debug,
                    first_rgb=prefixes[0]["rgb"],
                    second_rgb=prefix["rgb"],
                    first_depth=prefixes[0]["depth"],
                    second_depth=prefix["depth"],
                )
        for other in traces[1:]:
            if len(other) != len(traces[0]):
                prefix_equal = False
            for a, b in zip(traces[0], other):
                max_error = max(max_error, float(np.max(np.abs(a["pose"] - b["pose"]))))
                max_control_error = max(
                    max_control_error,
                    abs(a["control"].linear - b["control"].linear),
                    abs(a["control"].angular - b["control"].angular),
                )
                match, errors = self._sensor_compare(a, b)
                sensors_equal = sensors_equal and match
                for key, value in errors.items():
                    observed_sensor_errors[key] = max(observed_sensor_errors[key], value)
        passed = (
            prefix_equal
            and max_error <= tolerances["pose"]
            and max_control_error <= tolerances["control"]
            and sensors_equal
        )
        return {
            "passed": bool(passed),
            "prefix_equal": bool(prefix_equal),
            "max_pose_error": max_error,
            "max_control_error": max_control_error,
            "sensor_errors": observed_sensor_errors,
            "sensors_equal": bool(sensors_equal),
            "steps": steps,
            "backend": self.env.backend_name,
            "tolerances": tolerances,
            "method": "full_snapshot_and_reset_replay"
            if self.env.backend_name == "reference"
            else "reset_and_replay",
        }
