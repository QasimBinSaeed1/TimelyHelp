"""Audit saved pilot conditioning against fresh nominal prefixes; never rewrite shards.

Run four independent, disjoint parts with --gpu N --part N --parts 4. Only the
nominal prefix up to the configured snapshot times is simulated: no branches,
training, calibration, or evaluation. A failed comparison never authorizes reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from timelyhelp.config import digest, load_config, provenance, write_json
from timelyhelp.experiments import _close, _environment, _generation_shard, plan_generation_jobs
from timelyhelp.simulation.reference import FEATURE_NAMES
from timelyhelp.simulation.scenarios import Scenario


# Fixed absolute tolerances, chosen before observing any comparison results.
# Raw pose shares the existing native replay tolerance. Frozen normalized history
# features use the same conservative 1e-5 ceiling, including control features
# (stricter than the existing 1e-4 native control ceiling). There is no rtol.
FEATURE_ATOL = 1e-5
OBSERVATION_ATOL = 1e-5
TIME_ATOL = 1e-9
TIME_FIELDS = {
    "time",
    "remaining_time",
    "capture_time",
    "last_capture",
    "last_arrival",
    "expires_at",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_observation(saved, fresh, path="observation"):
    """Compare the complete public schema, recording numeric and categorical errors."""
    # Saved public observations crossed JSON serialization. Normalize tuple-valued
    # authored map rectangles through that same boundary before schema comparison.
    fresh = json.loads(json.dumps(fresh, allow_nan=False))
    mismatches, numeric_errors = [], {}

    def mismatch(field, old, new, reason, **details):
        mismatches.append({"field": field, "saved": old, "fresh": new, "reason": reason, **details})

    def visit(old, new, field):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                child = f"{field}.{key}"
                if key not in old or key not in new:
                    mismatch(child, old.get(key), new.get(key), "missing key")
                else:
                    visit(old[key], new[key], child)
        elif isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                mismatch(field, old, new, "list length differs")
            for index, (left, right) in enumerate(zip(old, new)):
                visit(left, right, f"{field}[{index}]")
        elif (
            isinstance(old, (int, float))
            and not isinstance(old, bool)
            and isinstance(new, (int, float))
            and not isinstance(new, bool)
        ):
            if not math.isfinite(old) or not math.isfinite(new):
                mismatch(field, str(old), str(new), "nonfinite numeric value")
                return
            error = abs(float(old) - float(new))
            numeric_errors[field] = error
            limit = TIME_ATOL if field.rsplit(".", 1)[-1] in TIME_FIELDS else OBSERVATION_ATOL
            if error > limit:
                mismatch(
                    field,
                    old,
                    new,
                    "absolute error exceeds fixed tolerance",
                    abs_error=error,
                    atol=limit,
                )
        elif type(old) is not type(new) or old != new:
            mismatch(field, old, new, "categorical value or type differs")

    visit(saved, fresh, path)
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "numeric_max_abs_errors": numeric_errors,
    }


def compare_history(saved, fresh):
    old, new = np.asarray(saved, dtype=float), np.asarray(fresh, dtype=float)
    result = {"passed": False, "saved_shape": list(old.shape), "fresh_shape": list(new.shape)}
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != len(FEATURE_NAMES):
        return result | {"error": "history shape differs from frozen feature schema"}
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        return result | {"error": "history contains nonfinite values"}
    differences = np.abs(old - new)
    maxima = np.max(differences, axis=0)
    mismatches = []
    for index in np.flatnonzero(maxima > FEATURE_ATOL):
        row = int(np.argmax(differences[:, index]))
        mismatches.append(
            {
                "field": f"history[{row}].{FEATURE_NAMES[index]}",
                "saved": float(old[row, index]),
                "fresh": float(new[row, index]),
                "abs_error": float(maxima[index]),
                "atol": FEATURE_ATOL,
            }
        )
    return result | {
        "passed": not mismatches,
        "max_abs_error": float(maxima.max(initial=0.0)),
        "feature_max_abs_errors": dict(zip(FEATURE_NAMES, maxima.tolist())),
        "mismatches": mismatches,
    }


def conditioning_groups(config, job, saved):
    """Reject incomplete candidates or inconsistent siblings before native replay."""
    if saved.get("job_id") != job["id"] or saved.get("complete") is not True:
        raise ValueError("Shard identity or completion flag does not match the planned episode")
    scenario = Scenario.from_dict(job["scenario"])
    expected_identity = {
        "seed": job["seed"],
        "controller": job["controller"],
        "split": job["split"],
        "layout": scenario.layout_id,
        "setting": scenario.setting,
        "family": scenario.family,
        "backend": config["backend"],
        "disturbances": list(scenario.disturbances),
    }
    grouped = {}
    generation = config["generation"]
    candidates = {
        (source, bridge, float(delay), replication)
        for source in generation["sources"]
        for bridge in generation["bridges"]
        for delay in generation["delays"]
        for replication in range(generation["replications"])
    }
    for row in saved["records"]:
        for key, expected in expected_identity.items():
            if row.get(key) != expected:
                raise ValueError(f"Saved row {key} differs: {row.get(key)!r} != {expected!r}")
        if not row.get("prearrival_audit", {}).get("passed"):
            raise ValueError("Saved row does not contain a passing prearrival audit")
        when = float(row["observation"]["time"])
        if not math.isfinite(when):
            raise ValueError("Saved snapshot time is nonfinite")
        grouped.setdefault(when, []).append(row)
    for when, rows in grouped.items():
        identities = [
            (row["source"], row["bridge"], float(row["delay"]), row["replication"]) for row in rows
        ]
        if len(identities) != len(candidates) or set(identities) != candidates:
            raise ValueError(f"Snapshot {when} has an incomplete or duplicated candidate grid")
        first = rows[0]
        for row in rows[1:]:
            if (
                row["snapshot_id"] != first["snapshot_id"]
                or row["history"] != first["history"]
                or row["observation"] != first["observation"]
            ):
                raise ValueError(f"Snapshot {when} changed conditioning between sibling branches")
    return grouped


def check_job(config, job, shard, progress):
    before = sha256(shard)
    saved = json.loads(shard.read_text())
    grouped = conditioning_groups(config, job, saved)
    env = None
    snapshots, errors, consumed = [], [], set()
    try:
        env = _environment(
            config, Scenario.from_dict(job["scenario"]), job["seed"], job["controller"]
        )
        generation = config["generation"]
        times = sorted(
            float(t) for t in generation["snapshot_times"][: generation["snapshots_per_episode"]]
        )
        for target in times:
            while not env.done and env.time < target - TIME_ATOL:
                env.advance("nominal")
            progress(job["id"], env.time)
            matches = [when for when in grouped if abs(when - env.time) <= TIME_ATOL]
            if env.done:
                if matches:
                    errors.append(
                        {
                            "reason": "fresh prefix terminated before a saved snapshot",
                            "time": env.time,
                        }
                    )
                continue
            if len(matches) != 1:
                errors.append(
                    {
                        "reason": "fresh nonterminal snapshot has no unique saved match",
                        "target": target,
                        "fresh_time": env.time,
                        "saved_times": sorted(grouped),
                    }
                )
                continue
            when = matches[0]
            consumed.add(when)
            row = grouped[when][0]
            history = compare_history(row["history"], env.history_features())
            observation = compare_observation(row["observation"], env.observe())
            snapshots.append(
                {
                    "time": when,
                    "snapshot_id": row["snapshot_id"],
                    "sibling_records": len(grouped[when]),
                    "passed": history["passed"] and observation["passed"],
                    "history": history,
                    "observation": observation,
                }
            )
        for when in sorted(set(grouped) - consumed):
            errors.append({"reason": "saved snapshot was not reproduced", "saved_time": when})
    finally:
        _close(env)
    after = sha256(shard)
    if before != after:
        errors.append({"reason": "original shard changed during audit"})
    return {
        "job_id": job["id"],
        "seed": job["seed"],
        "controller": job["controller"],
        "canonical_episode": job["canonical_episode"],
        "scenario": job["scenario"],
        "shard": str(shard.resolve()),
        "old_shard_sha256": before,
        "after_shard_sha256": after,
        "saved_work_sha256": saved["work_sha256"],
        "saved_gpu": saved.get("gpu"),
        "passed": not errors and all(snapshot["passed"] for snapshot in snapshots),
        "snapshots": snapshots,
        "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data", default="outputs/pilot/data")
    parser.add_argument(
        "--output", required=True, help="Independent audit directory, outside original data"
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="Bound this audit part; the result explicitly records incomplete coverage",
    )
    args = parser.parse_args(argv)
    if (
        args.gpu < 0
        or args.parts < 1
        or not 0 <= args.part < args.parts
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("GPU must be nonnegative; 0 <= part < parts; limit must be positive")
    os.environ["TIMELYHELP_GPU"] = str(args.gpu)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    config = load_config(args.config)
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    if output == data or data in output.parents:
        parser.error("Audit output must be outside the original data directory")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TIMELYHELP_NATIVE_CACHE"] = str(output / "native")
    if (output / "checks.json").exists() or (output / "status.json").exists():
        parser.error("Use a new output directory to retain previous audit evidence")
    saved_provenance = json.loads((data / "provenance.json").read_text())
    if (
        saved_provenance.get("config_sha256") != digest(config)
        or saved_provenance.get("config") != config
    ):
        raise ValueError("Saved generation configuration differs; prefix reuse cannot be validated")
    planned = [
        job | {"canonical_episode": index} for index, job in enumerate(plan_generation_jobs(config))
    ]
    complete, unavailable = [], []
    for job in planned:
        shard = _generation_shard(data, job)
        if shard.exists():
            saved = json.loads(shard.read_text())
            if saved.get("complete") is True:
                complete.append((job, shard))
                continue
        unavailable.append(job["id"])
    selected = [
        (job, shard)
        for job, shard in complete
        if job["canonical_episode"] % args.parts == args.part
    ]
    available_count = len(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
    state = {
        "status": "running",
        "pid": os.getpid(),
        "gpu": args.gpu,
        "part": args.part,
        "parts": args.parts,
        "started_unix_time": time.time(),
        "total_planned_episodes": len(planned),
        "complete_saved_episodes": len(complete),
        "unavailable_job_ids": unavailable,
        "available_in_part": available_count,
        "selected_in_part": len(selected),
        "limited": args.limit is not None,
        "completed": 0,
        "passed": 0,
        "failed": 0,
    }
    write_json(output / "provenance.json", provenance(config, stage="audit_saved_pilot_prefixes"))
    write_json(
        output / "audit_config.json",
        {
            "arguments": vars(args),
            "history_atol": FEATURE_ATOL,
            "observation_atol": OBSERVATION_ATOL,
            "time_atol": TIME_ATOL,
            "relative_tolerance": 0,
            "categorical_and_schema_comparison": "exact",
            "rationale": "Frozen conditioning features and raw public numeric evidence must reproduce within1e-5; native pose tolerance1e-5 and control tolerance1e-4 are not relaxed. Prefix-only audit does not revalidate old counterfactual outcomes.",
            "saved_provenance_sha256": sha256(data / "provenance.json"),
        },
    )

    def save_status(job_id=None, simulation_time=None):
        state.update(job=job_id, simulation_time=simulation_time, unix_time=time.time())
        write_json(output / "status.json", state)

    results = []
    save_status()
    try:
        for job, shard in selected:
            save_status(job["id"], 0.0)
            print(
                f"[gpu {args.gpu}] checking {job['id']} ({state['completed']}/{len(selected)})",
                flush=True,
            )
            try:
                result = check_job(config, job, shard, save_status)
            except Exception as error:
                result = {
                    "job_id": job["id"],
                    "shard": str(shard),
                    "old_shard_sha256": sha256(shard),
                    "passed": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            results.append(result)
            write_json(output / "episodes" / f"{job['canonical_episode']:04d}.json", result)
            state["completed"] += 1
            state["passed" if result["passed"] else "failed"] += 1
            save_status()
        maxima = {name: 0.0 for name in FEATURE_NAMES}
        for result in results:
            for snapshot in result.get("snapshots", []):
                for name, error in snapshot["history"].get("feature_max_abs_errors", {}).items():
                    maxima[name] = max(maxima[name], error)
        passed = state["failed"] == 0 and bool(selected)
        state.update(status="complete" if passed else "failed", finished_unix_time=time.time())
        write_json(
            output / "checks.json",
            {
                **state,
                "passed": passed,
                "full_part_coverage": len(selected) == available_count,
                "feature_max_abs_errors": maxima,
                "episodes": results,
                "old_shards_rewritten": False,
                "automatic_reuse_authorized": False,
                "scope": "raw saved conditioning versus fresh nominal prefixes only; no outcome revalidation",
            },
        )
        save_status()
        return 0 if passed else 1
    except BaseException as error:
        state.update(
            status="interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            finished_unix_time=time.time(),
        )
        save_status()
        raise
    finally:
        from timelyhelp.backends.native import _close_pool

        _close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
