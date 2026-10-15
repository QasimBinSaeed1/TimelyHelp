"""Bounded native W/Q audits using the unchanged pilot scenarios and mission horizon.

Only the common prefix and prearrival interval execute; no A branch, training,
full mission continuation, image correction, or tolerance relaxation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from timelyhelp.backends.native import _close_pool
from timelyhelp.config import load_config, provenance, write_json
from timelyhelp.counterfactual import CounterfactualRunner
from timelyhelp.experiments import _environment, plan_generation_jobs
from timelyhelp.simulation.scenarios import Scenario
from timelyhelp.types import Candidate


TIME_ATOL = 1e-8


def csv_values(value, convert=str):
    """Reject empty/duplicate lists instead of silently skipping requested cases."""
    pieces = [item.strip() for item in value.split(",")]
    if any(not piece for piece in pieces):
        raise ValueError("Lists cannot contain empty entries")
    result = [convert(piece) for piece in pieces]
    if not result or len(result) != len(set(result)):
        raise ValueError("Lists must be nonempty and contain unique values")
    return result


def tick_count(duration, dt):
    return max(0, math.ceil(duration / dt - 1e-9))


def compare_samples(runner, first, second):
    matched, sensors = runner._sensor_compare(first, second)
    pose = float(np.max(np.abs(first["pose"] - second["pose"])))
    control = max(
        abs(getattr(first["control"], name) - getattr(second["control"], name))
        for name in ("linear", "angular")
    )
    time_error = abs(first["time"] - second["time"])
    limits = runner._tolerances()
    return {
        "passed": bool(
            matched
            and pose <= limits["pose"]
            and control <= limits["control"]
            and time_error <= TIME_ATOL
        ),
        "pose_max": pose,
        "control_max": control,
        "time_error": time_error,
        "sensors_equal": bool(matched),
        "sensor_errors": sensors,
        "rgb_shapes": [list(first["rgb"].shape), list(second["rgb"].shape)],
        "depth_shapes": [list(first["depth"].shape), list(second["depth"].shape)],
    }


def save_failure(output, first, second):
    path = output / "first_failure.npz"
    if not path.exists():
        np.savez(
            path,
            first_rgb=first["rgb"],
            second_rgb=second["rgb"],
            first_depth=first["depth"],
            second_depth=second["depth"],
            first_pose=first["pose"],
            second_pose=second["pose"],
            first_time=first["time"],
            second_time=second["time"],
        )
    return str(path)


def branch(env, runner, snapshot, candidate, replication, query):
    trial = None
    try:
        trial = env.spawn_from_snapshot(snapshot)
        prefix = runner._sample(trial)
        trial.replication = replication
        request_id = trial.request(candidate, exact_delay=candidate.delay) if query else None
        request_prefix = runner._sample(trial)
        trace = []
        deadline = snapshot["time"] + candidate.delay
        while not trial.done and trial.time < deadline - 1e-9:
            trial.advance(candidate.bridge)
            trace.append(runner._sample(trial))
        return {
            "prefix": prefix,
            "request_prefix": request_prefix,
            "trace": trace,
            "request_id": request_id,
            "end_time": trial.time,
            "done": bool(trial.done),
            "collision": bool(trial.collision),
            "completion": bool(trial.completion),
        }
    finally:
        if trial is not None:
            trial.close()


def trace_contract(first, second, start, delay, dt, horizon):
    """Reject missing ticks, unequal termination, or an early nonterminal stop."""
    bounded_ticks = min(tick_count(delay, dt), tick_count(horizon - start, dt))
    failures = []
    if len(first["trace"]) != len(second["trace"]):
        failures.append("W/Q trace lengths differ")
    for key in ("done", "collision", "completion"):
        if first[key] != second[key]:
            failures.append(f"W/Q {key} flags differ")
    if abs(first["end_time"] - second["end_time"]) > TIME_ATOL:
        failures.append("W/Q final simulation times differ")
    for name, result in (("W", first), ("Q", second)):
        actual_ticks = len(result["trace"])
        expected_done = (
            result["collision"] or result["completion"] or result["end_time"] >= horizon - 1e-9
        )
        if result["done"] != expected_done:
            failures.append(f"{name} done flag disagrees with its terminal conditions")
        expected_ticks = bounded_ticks
        if result["done"] and result["end_time"] < start + bounded_ticks * dt - TIME_ATOL:
            # Collision/completion can legitimately terminate both branches early.
            expected_ticks = tick_count(result["end_time"] - start, dt)
            if not (result["collision"] or result["completion"]):
                failures.append(f"{name} terminated early without collision/completion")
        if actual_ticks != expected_ticks:
            failures.append(f"{name} has {actual_ticks} ticks; expected {expected_ticks}")
        if abs(result["end_time"] - (start + actual_ticks * dt)) > TIME_ATOL:
            failures.append(f"{name} final time does not match its tick count")
        for index, sample in enumerate(result["trace"], 1):
            if abs(sample["time"] - (start + index * dt)) > TIME_ATOL:
                failures.append(f"{name} tick {index} has an invalid simulation time")
    return {"passed": not failures, "bounded_expected_ticks": bounded_ticks, "failures": failures}


def check_pair(env, runner, snapshot, canonical, candidate, replication, output):
    w = branch(env, runner, snapshot, candidate, replication, False)
    q = branch(env, runner, snapshot, candidate, replication, True)
    checks = []
    comparisons = [
        ("canonical/W_prefix", canonical, w["prefix"]),
        ("canonical/Q_prefix", canonical, q["prefix"]),
        ("W/Q_prefix", w["prefix"], q["prefix"]),
        ("W/Q_after_request", w["request_prefix"], q["request_prefix"]),
    ]
    comparisons.extend(
        (f"tick_{index}", first, second)
        for index, (first, second) in enumerate(zip(w["trace"], q["trace"]), 1)
    )
    for label, first, second in comparisons:
        check = {"label": label, "time": first["time"], **compare_samples(runner, first, second)}
        if not check["passed"]:
            check["failure_arrays"] = save_failure(output, first, second)
        checks.append(check)
    contract = trace_contract(
        w, q, snapshot["time"], candidate.delay, env.config.dt, env.config.horizon
    )
    request_created = q["request_id"] is not None
    return {
        "candidate": candidate.to_dict(),
        "replication": replication,
        "passed": contract["passed"] and request_created and all(c["passed"] for c in checks),
        "request_created": request_created,
        "trace_contract": contract,
        "branches": {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"prefix", "request_prefix", "trace"}
            }
            | {"ticks": len(result["trace"])}
            for name, result in (("W", w), ("Q", q))
        },
        "comparisons": checks,
    }


def stationary_clock_check(capture_times, wrapper_times):
    """Compare each clock with itself; their origins/precision can differ."""
    first_label = next(iter(capture_times))
    native_start = capture_times[first_label]
    wrapper_start = wrapper_times[first_label]
    native_changes = {label: time - native_start for label, time in capture_times.items()}
    wrapper_changes = {label: time - wrapper_start for label, time in wrapper_times.items()}
    offsets = {label: time - wrapper_times[label] for label, time in capture_times.items()}
    native_max = max(abs(change) for change in native_changes.values())
    wrapper_max = max(abs(change) for change in wrapper_changes.values())
    return {
        "passed": native_max <= TIME_ATOL and wrapper_max <= TIME_ATOL,
        "native_capture_times": capture_times,
        "wrapper_times": wrapper_times,
        "native_minus_wrapper_offsets_s": offsets,
        "native_clock_changes_s": native_changes,
        "wrapper_clock_changes_s": wrapper_changes,
        "native_clock_change_max_s": native_max,
        "wrapper_clock_change_max_s": wrapper_max,
        "stationarity_time_atol_s": TIME_ATOL,
    }


def capture_interleave(env, runner, output):
    """Changing camera capture order must preserve stationary scientific sensing."""
    state = (env.time, env.pose.copy(), env.done, env.collision, env.completion)
    capture_times = {}
    wrapper_times = {}

    def capture(label, source, pose):
        frame = env._render(pose, source)
        capture_times[label] = float(frame.capture_time)
        wrapper_times[label] = float(env.time)
        return {
            **runner._sample(env),
            "rgb": frame.rgb,
            "depth": frame.depth,
            "dropped": frame.dropped,
        }

    first = capture("first", "onboard", env.pose)
    repeated = capture("repeated", "onboard", env.pose)
    capture("infrastructure", "infrastructure", np.asarray(env.geometry["camera_pose"]))
    interleaved = capture("interleaved", "onboard", env.pose)
    results = []
    for label, sample in (("repeated", repeated), ("interleaved", interleaved)):
        check = {"label": label, **compare_samples(runner, first, sample)}
        if not check["passed"]:
            check["failure_arrays"] = save_failure(output, first, sample)
        results.append(check)
    state_unchanged = bool(
        env.time == state[0]
        and np.array_equal(env.pose, state[1])
        and (env.done, env.collision, env.completion) == state[2:]
    )
    clocks = stationary_clock_check(capture_times, wrapper_times)
    return {
        "passed": state_unchanged and clocks["passed"] and all(c["passed"] for c in results),
        "physical_state_unchanged": state_unchanged,
        "clocks": clocks,
        "comparisons": results,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/pilot.yaml")
    result.add_argument("--episodes", help="Comma-separated generation job IDs")
    result.add_argument(
        "--episode", action="append", default=[], help="Repeatable generation job ID"
    )
    result.add_argument("--snapshot-times", default="0.5,1.5,2.5")
    result.add_argument("--delays", default="0.25,3")
    result.add_argument("--replications", default="0,2")
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--sources", help="Comma-separated sources; defaults to configured sources")
    result.add_argument("--bridges", help="Comma-separated bridges; defaults to configured bridges")
    result.add_argument("--capture-interleave", action="store_true")
    result.add_argument("--output", required=True)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if config["backend"] not in {"isaac", "gazebo"}:
            raise ValueError("This diagnostic requires a native simulator backend")
        episodes = (csv_values(args.episodes) if args.episodes else []) + args.episode
        episodes = episodes or ["generate/reactive/development/0000"]
        if len(episodes) != len(set(episodes)):
            raise ValueError("Episode IDs must be unique")
        jobs = {job["id"]: job for job in plan_generation_jobs(config)}
        if set(episodes) - jobs.keys():
            raise ValueError(f"Unknown generation jobs: {sorted(set(episodes) - jobs.keys())}")
        times = sorted(csv_values(args.snapshot_times, float))
        delays = csv_values(args.delays, float)
        replications = csv_values(args.replications, int)
        horizon = float(config.get("simulation", {}).get("horizon", 18.0))
        if any(not math.isfinite(t) or t < 0 or t >= horizon for t in times):
            raise ValueError("Snapshot times must be finite and inside the mission horizon")
        if any(not math.isfinite(d) or d < 0 for d in delays):
            raise ValueError("Delays must be finite and nonnegative")
        if args.repeats < 1 or any(r < 0 for r in replications):
            raise ValueError("Repeats must be positive and replications nonnegative")
        sources = csv_values(args.sources) if args.sources else config["generation"]["sources"]
        bridges = csv_values(args.bridges) if args.bridges else config["generation"]["bridges"]
        candidates = [Candidate(s, b, d) for s, b, d in itertools.product(sources, bridges, delays)]
        if "none" in sources:
            raise ValueError("A W/Q audit requires an actual query source")
        output = Path(args.output)
        if (output / "checks.json").exists() or (output / "first_failure.npz").exists():
            raise ValueError(
                "Choose a fresh output directory to preserve earlier diagnostic evidence"
            )
    except (ValueError, KeyError, OSError) as exc:
        argument_parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    run_provenance = provenance(config, stage="bounded_native_pairs")
    run_provenance.update(
        arguments=vars(args),
        command=sys.argv if argv is None else [__file__, *argv],
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=os.getenv("TIMELYHELP_GPU", "2"),
        full_missions_executed=False,
        training_executed=False,
        mission_horizon_unchanged=True,
    )
    write_json(output / "provenance.json", run_provenance)
    report = {"passed": False, "status": "running", "results": []}
    env = None
    try:
        for episode in episodes:
            job = jobs[episode]
            env = _environment(
                config, Scenario.from_dict(job["scenario"]), job["seed"], job["controller"]
            )
            runner = CounterfactualRunner(env)
            report["tolerances"] = runner._tolerances()
            for target_time in times:
                while not env.done and env.time < target_time - 1e-9:
                    env.advance("nominal")
                if env.done:
                    raise AssertionError(f"{episode} terminated before snapshot {target_time}")
                snapshot = env.snapshot()
                canonical = runner._sample(env)
                if abs(env.time - target_time) > TIME_ATOL:
                    raise ValueError(f"Snapshot {target_time} is not aligned to dt={env.config.dt}")
                if args.capture_interleave:
                    check = capture_interleave(env, runner, output)
                    row = {
                        "episode": episode,
                        "snapshot_time": env.time,
                        "kind": "capture_interleave",
                        **check,
                    }
                    report["results"].append(row)
                    write_json(output / "checks.json", report)
                    if not row["passed"]:
                        raise AssertionError("Stationary camera interleave audit failed")
                for repeat, replication, candidate in itertools.product(
                    range(args.repeats), replications, candidates
                ):
                    check = check_pair(
                        env, runner, snapshot, canonical, candidate, replication, output
                    )
                    row = {
                        "episode": episode,
                        "seed": job["seed"],
                        "controller": job["controller"],
                        "snapshot_time": env.time,
                        "repeat": repeat,
                        "kind": "prearrival_pair",
                        **check,
                    }
                    report["results"].append(row)
                    write_json(output / "checks.json", report)
                    print(
                        json.dumps(
                            {key: value for key, value in row.items() if key != "comparisons"}
                        ),
                        flush=True,
                    )
                    if not row["passed"]:
                        raise AssertionError(
                            f"Native W/Q audit failed: {episode}, snapshot={env.time}, {candidate}, replication={replication}"
                        )
            env.close()
            env = None
        report.update(passed=True, status="complete")
    except BaseException as exc:
        report.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            _close_pool()
            write_json(output / "checks.json", report)


if __name__ == "__main__":
    main()
