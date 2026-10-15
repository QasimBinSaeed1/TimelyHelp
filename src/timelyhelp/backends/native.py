"""Native simulator subprocesses with rendered RGB-D and reset-and-replay branches.

No native engine handle is copied or serialized. Every sibling continuation
rebuilds and executes the complete common prefix before applying its intervention.
"""

from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import uuid
import atexit
import numpy as np
from ..simulation import ReferenceSimulator, make_scenario
from ..types import CameraFrame, Control, Candidate, SimConfig

ROOT = Path(__file__).resolve().parents[3]

# Prespecified transfer shifts. They describe the evaluation protocol, not
# per-actor intentions or hidden reliability; policy observations never receive
# this privileged dynamics configuration.
NATIVE_ACTOR_BEHAVIOR = {
    "isaac": {
        "speed_multiplier": 1.0,
        "avoidance_strength": 0.25,
        "moving_robot_yield_factor": 0.75,
    },
    "gazebo": {
        "speed_multiplier": 1.1,
        "avoidance_strength": 0.35,
        "moving_robot_yield_factor": 0.85,
    },
}


def native_backend_protocol(backend):
    profile = NATIVE_ACTOR_BEHAVIOR[backend]
    return {
        "actor_speed_scale": profile["speed_multiplier"],
        "avoidance_gain": profile["avoidance_strength"],
        "moving_robot_yield_factor": profile["moving_robot_yield_factor"],
        "physics_engine": "PhysX" if backend == "isaac" else "DART",
        "renderer": "RTX RayTracedLighting" if backend == "isaac" else "Ogre2",
    }


def _native_actor_velocity(actor, robot_pose, robot_velocity, heading_noise, backend):
    profile = NATIVE_ACTOR_BEHAVIOR[backend]
    delta = actor["destination"] - actor["position"]
    distance = np.linalg.norm(delta)
    desired = profile["speed_multiplier"] * actor["speed"] * delta / max(distance, 1e-9)
    relative = actor["position"] - robot_pose[:2]
    robot_distance = np.linalg.norm(relative)
    if robot_distance < 1.6:
        desired += (
            profile["avoidance_strength"]
            * (1.6 - robot_distance)
            * relative
            / max(robot_distance, 1e-6)
        )
        if robot_velocity > 0.1:
            desired *= profile["moving_robot_yield_factor"]
    desired += 0.045 * heading_noise * np.array([-delta[1], delta[0]]) / max(distance, 1e-9)
    return desired if distance > 0.15 else np.zeros(2)


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    return value


class NativeDriver:
    def __init__(self, backend, cache=None, startup_timeout=240.0):
        if backend not in ("isaac", "gazebo"):
            raise ValueError("Unknown native backend")
        self.backend = backend
        self.identifier = uuid.uuid4().hex[:12]
        if cache is not None:
            cache_path = Path(cache)
        elif os.environ.get("TIMELYHELP_NATIVE_CACHE"):
            cache_path = (
                Path(os.environ["TIMELYHELP_NATIVE_CACHE"]) / f"{backend}_{self.identifier}"
            )
        else:
            cache_path = ROOT / "artifacts" / "native" / f"{backend}_{self.identifier}"
        self.cache = cache_path.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.socket_path = Path(tempfile.gettempdir()) / f"timelyhelp_{self.identifier}.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.socket_path))
        self.listener.listen(1)
        self.listener.settimeout(1.0)
        launcher = (
            ROOT / ".local/isaacsim/python.sh"
            if backend == "isaac"
            else ROOT / "scripts/gazebo_python.sh"
        )
        if not launcher.is_file():
            raise RuntimeError(
                f"Native {backend} launcher missing: {launcher}; run documented setup"
            )
        worker = Path(__file__).with_name(f"{backend}_worker.py")
        self.log_path = self.cache / "worker.log"
        self.log = self.log_path.open("w")
        env = os.environ.copy()
        env["TIMELYHELP_RUN_ID"] = self.identifier
        self.process = subprocess.Popen(
            [
                str(launcher),
                str(worker),
                "--socket",
                str(self.socket_path),
                "--cache",
                str(self.cache),
            ],
            cwd=ROOT,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        print(
            f"native {backend} process pid={self.process.pid} "
            f"TIMELYHELP_GPU={env.get('TIMELYHELP_GPU')} log={self.log_path}",
            flush=True,
        )
        self.connection = None
        self.channel = None
        self.closed = False
        deadline = time.monotonic() + startup_timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"{backend} worker exited {self.process.returncode}; log: {self.log_path}"
                    )
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    pass
            if self.connection is None:
                raise TimeoutError(
                    f"{backend} startup exceeded {startup_timeout}s; log: {self.log_path}"
                )
            self.connection.settimeout(startup_timeout)
            self.channel = self.connection.makefile("rw")
            ready = json.loads(self.channel.readline())
            if not ready.get("ready"):
                raise RuntimeError(f"Native startup failed: {ready}")
        except Exception:
            self.close()
            raise

    def rpc(self, command, data=None, timeout=120.0):
        if self.closed:
            raise RuntimeError("Native worker already closed")
        self.connection.settimeout(timeout)
        self.channel.write(
            json.dumps({"command": command, "data": _serial(data or {})}, allow_nan=False) + "\n"
        )
        self.channel.flush()
        line = self.channel.readline()
        if not line:
            raise RuntimeError(f"Native {self.backend} worker disconnected; log: {self.log_path}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"Native {self.backend}: {response['error']}\nLog: {self.log_path}")
        return response["result"]

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        channel = getattr(self, "channel", None)
        if channel is not None:
            try:
                channel.write(json.dumps({"command": "close"}) + "\n")
                channel.flush()
            except (OSError, ValueError):
                pass
            try:
                channel.close()
            except OSError:
                pass
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
        process = getattr(self, "process", None)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.listener.close()
        self.socket_path.unlink(missing_ok=True)
        if getattr(self, "log", None) is not None:
            self.log.close()


_DRIVER_POOL = {"isaac": [], "gazebo": []}
# A parent rollout and one serial counterfactual worker can both be reused at
# the next episode. Bound idle engines to avoid retaining additional GPU worlds.
# Parallel pilots keep this bound inside each GPU process; they do not start a
# second copy of the experiment.
_MAX_IDLE_DRIVERS = max(1, int(os.environ.get("TIMELYHELP_MAX_IDLE_DRIVERS", "2")))


def _close_pool():
    for drivers in _DRIVER_POOL.values():
        for driver in drivers:
            driver.close()
        drivers.clear()


atexit.register(_close_pool)


class NativeSimulator(ReferenceSimulator):
    # Declared replay tolerances; audits record the measured mismatch. Native
    # camera floating point differences never silently become exact-match claims.
    audit_tolerances = {
        "pose": 1e-5,
        "control": 1e-4,
        "rgb_max": 5.0,
        "rgb_changed_fraction": 0.005,
        "depth_max": 0.05,
        "depth_valid_fraction": 0.005,
    }

    @property
    def backend_protocol(self):
        return native_backend_protocol(self._native_backend)

    def __init__(self, scenario=None, seed=0, config=None, backend="gazebo", cache=None):
        self.backend_name = backend
        self._native_backend = backend
        self._native_cache = cache
        self._driver = (
            _DRIVER_POOL[backend].pop() if _DRIVER_POOL[backend] else NativeDriver(backend, cache)
        )
        self._engine_ready = False
        self._closed = False
        try:
            super().__init__(scenario, seed, config)
        except Exception:
            self.close()
            raise

    def _ensure_engine(self):
        if self._engine_ready:
            return
        config = self.config.to_dict()
        config["wheel_friction"] = 0.3 if "payload_dynamics" in self.scenario.disturbances else 0.7
        geometry = copy.deepcopy(self.geometry)
        if "source_occlusion" in self.scenario.disturbances:
            x = geometry["crossing"]
            side = geometry["side"]
            geometry["rectangles"].append(
                (x - 0.6, min(-side * 1.2, -side * 0.6), x + 0.6, max(-side * 1.2, -side * 0.6))
            )
        result = self._driver.rpc(
            "reset", {"geometry": geometry, "actors": self.actors, "config": config}, timeout=180.0
        )
        self.pose = np.asarray(result["pose"], float)
        self._engine_ready = True

    def _render(self, pose, source="onboard"):
        self._ensure_engine()
        result = self._driver.rpc("render", {"pose": pose, "source": source})
        if result.get("collision", False):
            self.collision = True
            self.completion = False
            self.done = True
        path = Path(result["path"])
        with np.load(path, allow_pickle=False) as saved:
            rgb = saved["rgb"].copy()
            depth = saved["depth"].copy()
            xy = saved["xy_map"].copy()
        path.unlink(missing_ok=True)
        # Native clipping planes sometimes encode empty rays as infinity or NaN.
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, self.config.sensor_range).astype(
            np.float32
        )
        native_pose = np.asarray(result["pose"], float)
        return CameraFrame(rgb, depth, native_pose, float(result["capture_time"]), xy_map=xy)

    def _physics_step(self, control):
        self._ensure_engine()
        c = self.config
        dt = c.dt
        old_pose = self.pose.copy()
        # Reactive behavior observes the actual current native robot state. Asking
        # changes neither the intention nor the exogenous actor random numbers.
        for i, actor in enumerate(self.actors):
            actor["velocity"] = _native_actor_velocity(
                actor,
                self.pose,
                self.velocity,
                self.normal("actor_heading", i),
                self._native_backend,
            )
            actor["position"] += actor["velocity"] * dt
        payload = "payload_dynamics" in self.scenario.disturbances
        limit = (c.acceleration if control.linear > self.velocity else c.braking) * (
            0.45 if payload else 1.0
        )
        target = float(
            np.clip(control.linear, self.velocity - limit * dt, self.velocity + limit * dt)
        )
        target = float(np.clip(target, 0, c.max_speed))
        result = self._driver.rpc(
            "step", {"control": [target, float(control.angular)], "actors": self.actors}
        )
        self.pose = np.asarray(result["pose"], float)
        self.velocity = float(result["velocity"][0])
        self.angular_velocity = float(result["velocity"][1])
        self.tick += 1
        self.time = self.tick * dt
        self.distance += float(np.linalg.norm(self.pose[:2] - old_pose[:2]))
        gaps = [
            np.linalg.norm(a["position"] - self.pose[:2]) - a["radius"] - c.robot_radius
            for a in self.actors
        ]
        for rect in self.geometry["rectangles"]:
            closest = np.clip(self.pose[:2], rect[:2], rect[2:])
            gaps.append(np.linalg.norm(self.pose[:2] - closest) - c.robot_radius)
        gap = min(gaps, default=10.0)
        near = gap < c.near_miss_distance
        if near and not self._near_contact:
            self.near_misses += 1
        self._near_contact = near
        if "collision" not in result:
            raise RuntimeError("Native worker omitted mandatory physics-contact collision label")
        if result["collision"]:
            self.collision = True
        self.completion = bool(
            not self.collision and np.linalg.norm(self.pose[:2] - self.goal) < 0.35
        )
        self.done = self.collision or self.completion or self.time >= c.horizon - 1e-9

    def snapshot(self):
        return copy.deepcopy(
            {key: value for key, value in self.__dict__.items() if key != "_driver"}
        )

    def reset_replay(self, snapshot):
        new = type(self)(
            snapshot["scenario"],
            snapshot["seed"],
            snapshot["config"],
            backend=snapshot["_native_backend"],
        )
        if snapshot["replication"] != 0:
            new.close()
            raise ValueError(
                "Native common prefix replication must be zero; vary replications after audited restoration"
            )
        for source, stats in snapshot["source_stats"].items():
            for key in (
                "reliability_prior",
                "response_prior_alpha",
                "response_prior_beta",
                "reliability_calibration_count",
                "reliability_calibration",
            ):
                if key in stats:
                    new.source_stats[source][key] = copy.deepcopy(stats[key])
        new.history = []
        new._record_history()
        events = snapshot["request_log"]
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
        error = float(np.max(np.abs(new.pose - snapshot["pose"])))
        if error > self.audit_tolerances["pose"]:
            new.close()
            raise AssertionError(
                f"{self.backend_name} reset-and-replay pose mismatch {error} exceeds tolerance"
            )
        new.admission_filter = snapshot.get("admission_filter")
        return new

    def spawn_from_snapshot(self, snapshot):
        return self.reset_replay(snapshot)

    def fork(self, snapshot=None):
        return self.spawn_from_snapshot(snapshot or self.snapshot())

    def restore(self, snapshot):
        replacement = self.reset_replay(snapshot)
        self.close()
        self.__dict__ = replacement.__dict__
        return self

    def close(self):
        driver = getattr(self, "_driver", None)
        if driver is None or getattr(self, "_closed", False):
            return
        self._closed = True
        if (
            not driver.closed
            and driver.process.poll() is None
            and len(_DRIVER_POOL[self._native_backend]) < _MAX_IDLE_DRIVERS
        ):
            _DRIVER_POOL[self._native_backend].append(driver)
        else:
            driver.close()


def backend_factory(backend="reference", scenario=None, seed=0, config=None, **kwargs):
    if backend == "reference":
        return ReferenceSimulator(scenario, seed, config)
    if backend in ("isaac", "gazebo"):
        return NativeSimulator(scenario, seed, config, backend, **kwargs)
    raise ValueError(f"Unknown backend {backend}")


make_backend = backend_factory


def native_smoke(backend, output, steps=3):
    """Bounded genuine physics, rendered-source, replay, pair and contact audit."""
    from ..config import provenance
    from ..counterfactual import CounterfactualRunner

    if steps < 1 or steps > 10:
        raise ValueError("Native smoke accepts one to ten ticks only")
    output = Path(output)
    destination = output if output.suffix == ".json" else output / "native_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    simulation_config = SimConfig(horizon=(steps + 5) * 0.1, camera_height=48)
    audit_config = {
        "mode": "smoke",
        "backend": backend,
        "steps": steps,
        "seed": 0,
        "simulation": simulation_config.to_dict(),
    }
    report = {
        "backend": backend,
        "passed": False,
        "steps": steps,
        "full_experiment": False,
        "backend_protocol": native_backend_protocol(backend),
        "provenance": provenance(audit_config, stage="native_smoke"),
    }
    env = None
    try:
        env = NativeSimulator(
            make_scenario(),
            seed=0,
            config=simulation_config,
            backend=backend,
        )
        initial_pose = env.pose.copy()
        initial_depth = env.observe()["depth_min"]
        for _ in range(steps):
            env.advance("nominal")
        report["movement"] = float(np.linalg.norm(env.pose[:2] - initial_pose[:2]))
        report["initial_front_depth"] = float(initial_depth)
        report["rgb_shape"] = list(env.frame.rgb.shape)
        report["depth_shape"] = list(env.frame.depth.shape)
        report["source_tracks"] = len(env.source_tracker.tracks)
        report["source_capture_time"] = float(env.source_frames[-1].capture_time)
        snapshot = env.snapshot()
        runner = CounterfactualRunner(env)
        report["replay_audit"] = runner.audit_restore(snapshot, steps=2)
        row = runner.run(snapshot, Candidate("infrastructure", "cautious", 0.2))
        report["paired_audit"] = row["prearrival_audit"]
        report["message_lifecycle"] = row["messages"]
        report["metrics"] = env.outcomes()
        # A separately marked oracle-only label audit introduces physical overlap;
        # it is not an autonomous policy episode or a performance result.
        env.actors[0]["position"] = env.pose[:2] + np.array([0.15, 0.0])
        env.actors[0]["destination"] = env.actors[0]["position"].copy()
        env.step(Control(0.0, 0.0))
        report["native_contact_detected"] = bool(env.collision)
        report["passed"] = bool(
            report["movement"] > 0.005
            and initial_depth > 0.35
            and report["source_tracks"] > 0
            and report["replay_audit"]["passed"]
            and report["paired_audit"]["passed"]
            and report["native_contact_detected"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        _close_pool()
    destination.write_text(json.dumps(_serial(report), indent=2, allow_nan=False) + "\n")
    return report
