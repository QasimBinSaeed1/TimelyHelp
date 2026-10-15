"""One genuine Isaac world per assigned GPU; no fitting or label generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from PIL import Image

from .baselines import BaselineConfig, make_baseline
from .backends.native import NativeSimulator, _close_pool
from .evaluation import run_episode
from .live import atomic_json, read_json
from .simulation.scenarios import make_scenario
from .types import SimConfig


class DemoStopped(Exception):
    pass


def write_jpeg(path, pixels):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(temporary, format="JPEG", quality=88)
    temporary.replace(path)


def depth_pixels(depth, sensor_range):
    """Actual depth, clipped at sensor range: near is yellow, far is blue."""
    depth = np.nan_to_num(np.asarray(depth), nan=sensor_range, posinf=sensor_range, neginf=0)
    scale = np.clip(depth / sensor_range, 0, 1)
    return np.stack((255 * (1 - scale), 220 * (1 - scale), 255 * scale), axis=-1).astype(np.uint8)


class DemoWorker:
    def __init__(self, gpu, slot, duration, output):
        self.gpu = gpu
        self.output = Path(output)
        self.directory = self.output / f"gpu_{gpu}"
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = [
            ("worksite", "worker_crossing"),
            ("worksite", "cart_junction"),
            ("worksite", "ambiguous_intent"),
            ("sidewalk", "worker_crossing"),
        ]
        setting, family = settings[slot % len(settings)]
        self.scenario = make_scenario("test", setting, family, slot % 2)
        self.config = SimConfig(horizon=duration, camera_width=96, camera_height=48, max_queries=6)
        self.run = 0
        self.environment = None
        self.last_capture = -1.0
        self.last_view = None
        self.last_camera = None
        self.started = time.monotonic()
        self.stopping = False

    def control(self):
        control = read_json(self.output / "control.json", {})
        if self.stopping or control.get("stop"):
            raise DemoStopped()
        return control

    def status(self, phase, telemetry=None, **extra):
        observation = self.environment.observe() if self.environment is not None else {}
        atomic_json(
            self.directory / "status.json",
            {
                "gpu": self.gpu,
                "phase": phase,
                "run": self.run,
                "setting": self.scenario.setting,
                "family": self.scenario.family,
                "policy": "always_query baseline",
                "backend": "Isaac Sim / PhysX / RTX",
                "horizon": self.config.horizon,
                "updated_unix": time.time(),
                "observation": observation,
                "telemetry": telemetry or {},
                "wall_elapsed": time.monotonic() - self.started,
                **extra,
            },
        )

    def capture(self):
        control = self.control()
        env = self.environment
        camera = {key: control.get(key) for key in ("view", "azimuth", "elevation", "distance")}
        reply = env._driver.rpc("viewer_frame", dict(camera, setting=self.scenario.setting))
        path = Path(reply["path"])
        try:
            with np.load(path) as archive:
                write_jpeg(self.directory / "world.jpg", archive["rgb"])
        finally:
            path.unlink(missing_ok=True)
        write_jpeg(self.directory / "onboard.jpg", env.frame.rgb)
        write_jpeg(self.directory / "infrastructure.jpg", env.source_frames[-1].rgb)
        write_jpeg(
            self.directory / "depth.jpg", depth_pixels(env.frame.depth, env.config.sensor_range)
        )
        self.last_capture = env.time
        self.last_view = control.get("view", "overview")
        self.last_camera = camera
        atomic_json(
            self.directory / "frames.json",
            {
                "world_capture_time": reply["capture_time"],
                "onboard_capture_time": env.frame.capture_time,
                "infrastructure_capture_time": env.source_frames[-1].capture_time,
                "depth_sensor_range": env.config.sensor_range,
                "view": self.last_view,
                "world_is_privileged_human_diagnostic": True,
                "presentation": reply.get("presentation"),
                "sensor_scene_restored": reply.get("sensor_scene_restored", False),
                "presentation_assets": reply.get("presentation_assets", {}),
            },
        )

    def tick(self, env, telemetry):
        if env.time - self.last_capture >= 0.29 or env.done:
            self.capture()
        self.status("running", telemetry)
        # Pace at no faster than wall clock; report achieved speed rather than
        # promising real-time performance when native rendering is slower.
        wait_until = self.started + env.time
        while True:
            control = self.control()
            if control.get("paused"):
                before = time.monotonic()
                self.status("paused", telemetry)
                while self.control().get("paused"):
                    if self.camera_changed(self.control()):
                        self.capture()
                    time.sleep(0.1)
                self.started += time.monotonic() - before
                self.status("running", telemetry)
                wait_until = self.started + env.time
            if time.monotonic() >= wait_until:
                break
            time.sleep(min(0.05, wait_until - time.monotonic()))

    def execute(self):
        self.status("starting")
        self.environment = NativeSimulator(
            self.scenario,
            seed=1700 + self.gpu,
            config=self.config,
            backend="isaac",
            cache=self.directory / "native",
        )
        self.capture()
        self.status("ready")
        while True:
            control = self.control()
            if self.camera_changed(control):
                self.capture()
            if control.get("run", 0) > self.run:
                if self.run:
                    self.environment.close()
                    self.environment = NativeSimulator(
                        self.scenario,
                        seed=1700 + self.gpu,
                        config=self.config,
                        backend="isaac",
                        cache=self.directory / "native",
                    )
                self.run = control["run"]
                self.started = time.monotonic()
                self.last_capture = -1
                self.status("running")
                policy = make_baseline("always_query", BaselineConfig(sim=self.config))
                outcomes = run_episode(self.environment, policy, step_callback=self.tick)
                atomic_json(
                    self.directory / f"mission_{self.run}.json",
                    {
                        "kind": "visual_smoke_only",
                        "gpu": self.gpu,
                        "seed": 1700 + self.gpu,
                        "scenario": asdict(self.scenario),
                        "simulation": asdict(self.config),
                        "policy": "always_query",
                        "results_are_smoke": True,
                        "training_executed": False,
                        "outcomes": outcomes,
                    },
                )
                self.capture()
                self.status("done", outcomes=outcomes)
            time.sleep(0.1)

    def camera_changed(self, control):
        return {
            key: control.get(key) for key in ("view", "azimuth", "elevation", "distance")
        } != self.last_camera

    def close(self):
        if self.environment is not None:
            self.environment.close()
        _close_pool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker = DemoWorker(args.gpu, args.slot, args.duration, args.output)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: setattr(worker, "stopping", True))
    try:
        worker.execute()
    except DemoStopped:
        worker.status("stopped")
    except Exception as error:
        traceback.print_exc()
        worker.status("error", error=str(error))
        return 1
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
