"""Short native audit: detailed viewer must not change sensors or mission behavior."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from timelyhelp.baselines import BaselineConfig, make_baseline
from timelyhelp.backends.native import NativeSimulator, _close_pool
from timelyhelp.evaluation import run_episode
from timelyhelp.simulation.scenarios import make_scenario
from timelyhelp.types import SimConfig


def capture(env, output, view):
    before = env._render(env.pose, "onboard")
    before_pose = env.pose.copy()
    before_time = env.time
    reply = env._driver.rpc("viewer_frame", dict(view=view, setting=env.scenario.setting))
    path = Path(reply["path"])
    with np.load(path) as frame:
        Image.fromarray(frame["rgb"]).save(output / f"{view}.png")
    path.unlink()
    after = env._render(env.pose, "onboard")
    errors = {
        "rgb_max": float(np.max(np.abs(before.rgb.astype(float) - after.rgb))),
        "rgb_changed_fraction": float(np.mean(np.any(before.rgb != after.rgb, axis=2))),
        "depth_max": float(np.max(np.abs(before.depth - after.depth))),
        "pose_max": float(np.max(np.abs(env.pose - before_pose))),
        "time_changed": env.time != before_time,
        "sensor_scene_restored": reply["sensor_scene_restored"],
    }
    limits = env.audit_tolerances
    assert errors["rgb_max"] <= limits["rgb_max"], errors
    assert errors["rgb_changed_fraction"] <= limits["rgb_changed_fraction"], errors
    assert errors["depth_max"] <= limits["depth_max"], errors
    assert errors["pose_max"] == 0 and not errors["time_changed"], errors
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/presentation_audit")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = SimConfig(horizon=1.2, camera_width=96, camera_height=48)
    scenario = make_scenario("test", "worksite", "worker_crossing")
    trajectories = []
    results = []
    checks = {}
    env = None
    try:
        for enabled in (False, True):
            env = NativeSimulator(
                scenario, seed=1700, config=config, backend="isaac", cache=output / "native"
            )
            if enabled:
                for view in ("follow", "overview", "top", "orbit"):
                    checks[view] = capture(env, output, view)
            rows = []

            def observe(environment, telemetry):
                if enabled and telemetry["step"] % 3 == 0:
                    checks[f"step_{telemetry['step']}"] = capture(environment, output, "follow")
                rows.append(
                    {
                        "pose": environment.pose.copy(),
                        "history": environment.history_features().copy(),
                        "rgb": environment.frame.rgb.copy(),
                        "depth": environment.frame.depth.copy(),
                    }
                )

            result = run_episode(
                env,
                make_baseline("always_query", BaselineConfig(sim=config)),
                step_callback=observe,
            )
            for key in ("selection_seconds_mean", "selection_seconds_max"):
                result.pop(key)
            results.append(result)
            trajectories.append(rows)
            env.close()
            env = None
        assert results[0] == results[1], "Inspection changed mission outcomes"
        errors = {
            key: max(
                float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
                for a, b in zip(*trajectories)
            )
            for key in ("pose", "history", "rgb", "depth")
        }
        assert errors["pose"] <= 1e-5 and errors["history"] <= 1e-5, errors
        assert errors["rgb"] <= 5 and errors["depth"] <= 0.05, errors
        (output / "checks.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "capture_checks": checks,
                    "paired_episode_max_errors": errors,
                    "identical_outcomes": True,
                    "native_tolerances_unchanged": True,
                    "full_pilot_executed": False,
                },
                indent=2,
            )
        )
        print(json.dumps(errors, indent=2))
    finally:
        if env is not None:
            env.close()
        _close_pool()


if __name__ == "__main__":
    main()
