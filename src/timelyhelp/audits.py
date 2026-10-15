"""Executable methodological audits and explicitly diagnostic pilot measurements.

Privileged branch outcomes are accepted here only for labels and post hoc audits.
None of the diagnostic oracle outputs is exposed to deployable observations.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .learning import (
    Candidate,
    PairedDataset,
    predict_useful_window,
    restricted_completion_time,
    validate_splits,
)
from .simulation import FEATURE_NAMES

OBSERVATION_KEYS = {
    "time",
    "pose",
    "velocity",
    "angular_velocity",
    "goal",
    "tracks",
    "depth_min",
    "depth_left",
    "depth_right",
    "dropout",
    "sources",
    "crossings",
    "map_obstacles",
    "blind_zone_uncertainty",
    "intent_beliefs",
    "remaining_time",
    "remaining_queries",
    "pending",
    "messages",
}
SOURCE_KEYS = {
    "coverage",
    "latencies",
    "latency_samples",
    "response_fraction",
    "status",
    "last_capture",
    "last_arrival",
    "reliability_prior",
    "camera_pose",
    "fov",
    "range",
    "advertised_latency",
    "available",
}
SOURCE_KEYS |= {"reliability_calibration", "reliability_calibration_count"}
TRACK_KEYS = {"position", "velocity", "covariance", "capture_time", "kind", "provenance", "intent"}
MESSAGE_KEYS = {
    "request_id",
    "source",
    "question",
    "capture_time",
    "arrival_time",
    "payload",
    "admitted",
    "reason",
}
PENDING_KEYS = {"request_id", "source", "reason", "cancelled", "expires_at"}
PAYLOAD_KEYS = {"tracks", "answer", "crossing"}
INTENT_KEYS = {"crossing", "answer", "capture_time", "arrival_time"}
FORBIDDEN_KEYS = {
    "actor_id",
    "actor_ids",
    "object_id",
    "object_ids",
    "simulator_id",
    "uid",
    "ground_truth",
    "groundtruth",
    "true_state",
    "hidden_state",
    "hidden_source_error",
    "true_reliability",
    "corrupted",
    "corruption",
    "corruption_label",
    "future_trajectory",
    "future_trajectories",
    "future_path",
    "actual_latency",
    "true_latency_draw",
}


def _check_keys(value: Mapping, allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Nonpublic fields at {path}: {sorted(unknown)}")


def _walk_public(value: Any, path: str = "observation") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Privileged field leaked at {path}.{key}")
            _walk_public(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list, np.ndarray)):
        for i, child in enumerate(value):
            _walk_public(child, f"{path}[{i}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"Nonfinite public numeric value at {path}")


def audit_public_observation(
    observation: Mapping[str, Any],
    history: np.ndarray | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    """Strict schema/timestamp audit; a schema cannot prove sensor provenance alone."""
    _walk_public(observation)
    _check_keys(observation, OBSERVATION_KEYS, "observation")
    now = float(observation["time"])
    if now < 0:
        raise ValueError("Observation time cannot be negative")
    for source, metadata in observation.get("sources", {}).items():
        if source not in ("infrastructure", "supervisor"):
            raise ValueError("Unknown public information source")
        _check_keys(metadata, SOURCE_KEYS, f"sources.{source}")
        for timestamp in ("last_capture", "last_arrival"):
            value = metadata.get(timestamp)
            if value is not None and value > now + 1e-8:
                raise ValueError("Future source timestamp leaked")
        for latency in metadata.get("latencies", metadata.get("latency_samples", [])):
            if latency < 0:
                raise ValueError("Negative measured source delay")
        for probability in ("response_fraction", "reliability_prior"):
            if probability in metadata and not 0 <= metadata[probability] <= 1:
                raise ValueError("Invalid advertised/measured probability")
    for track in observation.get("tracks", []):
        _check_keys(track, TRACK_KEYS, "tracks")
        if track["capture_time"] > now + 1e-8:
            raise ValueError("Future track capture leaked")
    for pending in observation.get("pending", []):
        _check_keys(pending, PENDING_KEYS, "pending")
    for message in observation.get("messages", []):
        _check_keys(message, MESSAGE_KEYS, "messages")
        if message["capture_time"] > now + 1e-8 or message["arrival_time"] > now + 1e-8:
            raise ValueError("Unarrived payload/timestamp leaked")
        _check_keys(message["payload"], PAYLOAD_KEYS, "messages.payload")
        for track in message["payload"].get("tracks", []):
            _check_keys(track, TRACK_KEYS, "messages.payload.tracks")
            if track["capture_time"] > message["capture_time"] + 1e-8:
                raise ValueError("Track captured after packet source capture")
    for intent in observation.get("intent_beliefs", []):
        _check_keys(intent, INTENT_KEYS, "intent_beliefs")
        if intent["capture_time"] > now + 1e-8 or intent["arrival_time"] > now + 1e-8:
            raise ValueError("Unarrived intent evidence leaked")
    if any(name.lower() in FORBIDDEN_KEYS for name in feature_names):
        raise ValueError("Privileged learned feature name")
    if history is not None:
        h = np.asarray(history)
        if h.ndim != 2 or h.shape[-1] != len(feature_names) or not np.isfinite(h).all():
            raise ValueError("Invalid observation-history schema")
    return {
        "passed": True,
        "root_fields": sorted(observation),
        "feature_names": list(feature_names),
        "sensor_provenance": "Schema checked; rendered detector and source code provenance require separate audit",
        "privileged_future_fields": "absent",
        "received_packets": len(observation.get("messages", [])),
    }


def _record_key(record: Mapping[str, Any]) -> tuple:
    return (
        record["layout"],
        record["rollout_id"],
        record["snapshot_id"],
        record["source"],
        record["bridge"],
        float(record["delay"]),
    )


def audit_controlled_labels(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check decomposition, no-answer identity, paired common controls and reuse."""
    records = list(records)
    if not records:
        raise ValueError("No controlled continuations supplied")
    validate_splits(records)
    seen, actual_a, actual_w, actual_q, audited = set(), {}, {}, {}, 0
    signs = {"beneficial": 0, "neutral": 0, "harmful": 0}
    for record in records:
        Candidate(record["source"], record["bridge"], record["delay"])
        key = _record_key(record) + (int(record.get("replication", 0)),)
        if key in seen:
            raise ValueError("Duplicate controlled continuation")
        seen.add(key)
        outcomes = record["outcomes"]
        a, w, q = (float(outcomes[b]["cost"]) for b in ("A", "W", "Q"))
        info, motion = float(record["info"]), float(record["motion"])
        if (
            not np.isfinite([a, w, q, info, motion]).all()
            or not np.isclose(info, w - q)
            or not np.isclose(motion, a - w)
        ):
            raise ValueError("Invalid controlled cost decomposition")
        if not np.isclose(info + motion, a - q):
            raise ValueError("Total effect identity failed")
        if record["source"] == "none" and not np.isclose(w, q):
            raise ValueError("No-query Q differs from W")
        signs["beneficial" if info > 1e-8 else "harmful" if info < -1e-8 else "neutral"] += 1
        base = (record["layout"], record["rollout_id"], record["snapshot_id"], key[-1])
        motion_key = base + (record["bridge"], float(record["delay"]))
        for table, branch_key, outcome in (
            (actual_a, base, outcomes["A"]),
            (actual_w, motion_key, outcomes["W"]),
            (actual_q, key, outcomes["Q"]),
        ):
            encoded = json.dumps(outcome, sort_keys=True)
            if branch_key in table and table[branch_key] != encoded:
                raise ValueError("Reused branch has inconsistent outcomes")
            table[branch_key] = encoded
        audit = record.get("prearrival_audit")
        if audit is not None:
            tolerances = audit.get("tolerances", {"pose": 1e-10, "control": 1e-10})
            if (
                not audit.get("passed", False)
                or audit.get("max_pose_error", 0) > float(tolerances.get("pose", 1e-10))
                or audit.get("max_control_error", 0) > float(tolerances.get("control", 1e-10))
            ):
                raise ValueError("Prearrival motion/sensor audit failed")
            audited += 1
        if "observation" in record:
            audit_public_observation(record["observation"], np.asarray(record["history"]))
        for branch in ("A", "W", "Q"):
            outcome = outcomes[branch]
            if outcome["collision"] and outcome["completion"]:
                raise ValueError("Collision and successful completion cannot coexist")
            if not np.isfinite(outcome["completion_time"]) or outcome["completion_time"] < 0:
                raise ValueError("Failed/time-limited branch lacks an elapsed-time target")
    return {
        "passed": True,
        "label_rows": len(records),
        "signed_rows": signs,
        "naive_three_branch_interactions": 3 * len(records),
        "distinct_branch_interactions": len(actual_a) + len(actual_w) + len(actual_q),
        "A_interactions": len(actual_a),
        "W_interactions": len(actual_w),
        "Q_interactions": len(actual_q),
        "prearrival_audited_rows": audited,
        "prearrival_status": "verified"
        if audited == len(records)
        else "partially audited or not supplied",
        "prearrival_tolerances": [
            r["prearrival_audit"].get("tolerances", {"pose": 1e-10, "control": 1e-10})
            for r in records
            if "prearrival_audit" in r
        ],
        "failures_retained": sum(
            not r["outcomes"][b]["completion"] for r in records for b in ("A", "W", "Q")
        ),
        "rescue_filter": False,
    }


def monte_carlo_summary(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    audit_controlled_labels(records)
    groups = defaultdict(list)
    for record in records:
        groups[_record_key(record)].append(record)
    summaries = []
    for key, rows in sorted(groups.items()):
        effects = np.asarray([[r["info"], r["motion"]] for r in rows], dtype=float)
        n = len(rows)
        stderr = (effects.std(axis=0, ddof=1) / np.sqrt(n)).tolist() if n > 1 else None
        summaries.append(
            {
                "layout": key[0],
                "rollout_id": key[1],
                "snapshot_id": key[2],
                "source": key[3],
                "bridge": key[4],
                "delay": key[5],
                "replications": n,
                "information_mean": float(effects[:, 0].mean()),
                "motion_mean": float(effects[:, 1].mean()),
                "effect_standard_error": stderr,
                "sampling_uncertainty_identified": n > 1,
                "harmful_fraction": float(np.mean(effects[:, 0] < -1e-8)),
                "neutral_fraction": float(np.mean(np.abs(effects[:, 0]) <= 1e-8)),
                "outcomes": {
                    b: {
                        **{
                            metric: float(np.mean([r["outcomes"][b][metric] for r in rows]))
                            for metric in ("cost", "collision", "completion")
                        },
                        "restricted_completion_time": float(
                            np.mean([restricted_completion_time(r, b) for r in rows])
                        ),
                        "elapsed_time": float(
                            np.mean(
                                [
                                    r["outcomes"][b].get(
                                        "time", r["outcomes"][b]["completion_time"]
                                    )
                                    for r in rows
                                ]
                            )
                        ),
                    }
                    for b in ("A", "W", "Q")
                },
            }
        )
    return summaries


def pilot_diagnostics(
    records: Iterable[Mapping[str, Any]],
    query_penalty: float = 0.03,
    minimum_material_effect: float = 0.05,
) -> dict[str, Any]:
    """Privileged oracle headroom/sign curves; never a deployable baseline result."""
    records = list(records)
    audit = audit_controlled_labels(records)
    summaries = monte_carlo_summary(records)
    snapshots, curves = defaultdict(list), defaultdict(list)
    for summary in summaries:
        snapshots[(summary["layout"], summary["rollout_id"], summary["snapshot_id"])].append(
            summary
        )
        curves[
            (
                summary["layout"],
                summary["rollout_id"],
                summary["snapshot_id"],
                summary["source"],
                summary["bridge"],
            )
        ].append(summary)
    oracles, curve_reports = [], []
    for key, rows in snapshots.items():
        nominal = rows[0]["outcomes"]["A"]["cost"]
        best_control = min(row["outcomes"]["W"]["cost"] for row in rows)
        best_query = min(
            rows,
            key=lambda row: row["outcomes"]["Q"]["cost"]
            + query_penalty * (row["source"] != "none"),
        )
        query_cost = best_query["outcomes"]["Q"]["cost"] + query_penalty * (
            best_query["source"] != "none"
        )
        oracle = min(nominal, best_control, query_cost)
        oracles.append(
            {
                "layout": key[0],
                "rollout_id": key[1],
                "snapshot_id": key[2],
                "nominal_cost": nominal,
                "best_noquery_cost": min(nominal, best_control),
                "oracle_cost": oracle,
                "gap_over_noquery": min(nominal, best_control) - oracle,
                "best_query": {k: best_query[k] for k in ("source", "bridge", "delay")},
            }
        )
    for key, rows in curves.items():
        rows = sorted(rows, key=lambda row: row["delay"])
        values = np.asarray([row["information_mean"] for row in rows])
        curve_reports.append(
            {
                "layout": key[0],
                "rollout_id": key[1],
                "snapshot_id": key[2],
                "source": key[3],
                "bridge": key[4],
                "delays": [row["delay"] for row in rows],
                "information": values.tolist(),
                "motion": [row["motion_mean"] for row in rows],
                "changes_sign": bool((values > 1e-8).any() and (values < -1e-8).any()),
                "material_delay_dependence": bool(np.ptp(values) >= minimum_material_effect),
            }
        )
    mean_gap = float(np.mean([row["gap_over_noquery"] for row in oracles]))
    return {
        "role": "privileged pilot diagnostic; no performance claim",
        "audit": audit,
        "snapshot_count": len(oracles),
        "oracle_snapshots": oracles,
        "benefit_curves": curve_reports,
        "oracle_mean_gap_over_strong_noquery": mean_gap,
        "positive_motion_matched_information_fraction": float(
            np.mean([r["information_mean"] > 0 for r in summaries])
        ),
        "delay_sign_change_fraction": float(np.mean([r["changes_sign"] for r in curve_reports])),
        "central_hypothesis_supported_by_this_pilot": False,
        "interpretation": "Predictability and advantage over the closed-loop delay-aware baseline require held-out tests; negligible oracle gap is negative evidence",
    }


def _confidence(observation: Mapping[str, Any]) -> float:
    tracks = observation.get("tracks", [])
    variance = float(np.mean([t["covariance"] for t in tracks])) if tracks else 1.0
    return float(
        np.clip(1 / (1 + variance) * (1 - observation.get("blind_zone_uncertainty", 0)), 0, 1)
    )


def matched_observable_scenes(
    records: Iterable[Mapping[str, Any]],
    max_pairs: int = 10,
    confidence_tolerance: float = 1e-5,
    query_penalty: float = 0.03,
) -> dict[str, Any]:
    """Explain equal-confidence decisions using public features; identify aliasing."""
    records = list(records)
    dataset = PairedDataset.from_records(records)
    snapshots = defaultdict(list)
    raw = {(r["layout"], str(r["rollout_id"]), str(r["snapshot_id"])): r for r in records}
    for i, m in enumerate(dataset.metadata):
        snapshots[(m["layout"], str(m["rollout_id"]), str(m["snapshot_id"]))].append(i)
    scenes = []
    for key, ids in snapshots.items():
        best = min(
            ids,
            key=lambda i: dataset.metadata[i]["mean_costs"][2]
            + query_penalty * (dataset.candidates[i].source != "none"),
        )
        h = dataset.histories[best]
        scene = {
            "key": key,
            "history": h,
            "choice": dataset.candidates[best],
            "confidence": _confidence(raw[key].get("observation", {})),
        }
        scenes.append(scene)
    equal_confidence, aliases = [], []
    for first, second in combinations(scenes, 2):
        if first["choice"] == second["choice"]:
            continue
        difference = np.max(np.abs(first["history"] - second["history"]), axis=0)
        explanation = [
            {
                "feature": FEATURE_NAMES[i],
                "max_history_difference": float(difference[i]),
                "first_current": float(first["history"][-1, i]),
                "second_current": float(second["history"][-1, i]),
            }
            for i in np.flatnonzero(difference > 1e-7)
        ]
        pair = {
            "first_snapshot": list(first["key"]),
            "second_snapshot": list(second["key"]),
            "first_choice": first["choice"].to_dict(),
            "second_choice": second["choice"].to_dict(),
            "first_confidence": first["confidence"],
            "second_confidence": second["confidence"],
            "accessible_differences": explanation,
        }
        if not explanation and len(aliases) < max_pairs:
            aliases.append(
                pair
                | {
                    "interpretation": "Observational aliasing; incompatible oracle outcomes are not predictably distinct assistance decisions"
                }
            )
        elif (
            abs(first["confidence"] - second["confidence"]) <= confidence_tolerance
            and len(equal_confidence) < max_pairs
        ):
            equal_confidence.append(pair)
        if len(aliases) >= max_pairs and len(equal_confidence) >= max_pairs:
            break
    return {
        "identical_confidence_pairs": equal_confidence,
        "identical_observation_aliases": aliases,
        "comparison": "Oracle Q and no-query W choices with declared query penalty from conditional Monte Carlo labels, post hoc only",
        "query_penalty": query_penalty,
        "missing_observation_records": sum("observation" not in r for r in records),
    }


def window_prediction_grid(
    model,
    dataset: PairedDataset,
    delays: Sequence[float],
    risk_limit: float = 0.1,
    uncertainty_weight: float = 1.0,
) -> list[dict[str, Any]]:
    result, seen = [], set()
    for i, metadata in enumerate(dataset.metadata):
        key = (metadata["layout"], metadata["rollout_id"], metadata["snapshot_id"])
        if key in seen:
            continue
        seen.add(key)
        for source in ("infrastructure", "supervisor"):
            for bridge in ("cautious", "hold"):
                window = predict_useful_window(
                    model,
                    dataset.histories[i],
                    source,
                    bridge,
                    delays,
                    risk_limit,
                    uncertainty_weight,
                )
                result.append(
                    {
                        **{
                            k: metadata[k] for k in ("layout", "rollout_id", "snapshot_id", "split")
                        },
                        "source": source,
                        "bridge": bridge,
                        **window.to_dict(),
                    }
                )
    return result


def process_tree_memory(worker_pid: int | None) -> dict[str, Any]:
    """Sample the native worker and its current Linux process descendants.

    These are privileged instrumentation identifiers, never policy features.
    Per-process RSS includes shared pages; summed high-water marks occurred at
    potentially different times and are not a simultaneous process-tree peak.
    """
    scope = (
        "Native worker and current descendants, including launcher/Xvfb when present; "
        "RSS includes shared pages. Sum of per-process high-water marks is not a "
        "simultaneous aggregate peak; excludes unrelated jobs and parent learner."
    )
    if worker_pid is None:
        return {
            "available": False,
            "worker_pid": None,
            "processes": [],
            "reason": "No separate native worker process",
            "scope": scope,
        }
    if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
        raise ValueError("Native worker PID must be a positive integer")
    pending, seen, measurements = [worker_pid], set(), []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        directory = Path("/proc") / str(pid)
        try:
            status = dict(
                line.split(":", 1)
                for line in (directory / "status").read_text().splitlines()
                if ":" in line
            )
        except (OSError, ValueError):
            continue  # A short-lived worker descendant may exit between reads.

        def bytes_from_kib(key):
            value = status.get(key)
            return None if value is None else int(value.split()[0]) * 1024

        measurements.append(
            {
                "pid": pid,
                "parent_pid": int(status["PPid"]),
                "name": status["Name"].strip(),
                "rss_bytes": bytes_from_kib("VmRSS"),
                "high_water_rss_bytes": bytes_from_kib("VmHWM"),
            }
        )
        # Children can be created by any worker thread. Read every task's child
        # list rather than assuming the main thread launched every subprocess.
        for children in (directory / "task").glob("*/children"):
            try:
                pending.extend(int(child) for child in children.read_text().split())
            except (OSError, ValueError):
                continue
    measurements.sort(key=lambda entry: entry["pid"])
    resident = [entry["rss_bytes"] for entry in measurements if entry["rss_bytes"] is not None]
    maxima = [
        entry["high_water_rss_bytes"]
        for entry in measurements
        if entry["high_water_rss_bytes"] is not None
    ]
    return {
        "available": bool(resident),
        "worker_pid": worker_pid,
        "processes": measurements,
        "sampled_tree_rss_bytes": sum(resident) if resident else None,
        "sum_process_high_water_rss_bytes": sum(maxima) if maxima else None,
        "reason": None if resident else "Worker exited or Linux /proc memory counters unavailable",
        "scope": scope,
        "measurement": "Linux /proc status VmRSS/VmHWM in bytes",
    }


def profile_resources(
    env,
    model=None,
    *,
    steps: int = 5,
    candidates: Sequence[Candidate] | None = None,
    prediction_repeats: int = 10,
) -> dict[str, Any]:
    """Measure short real stepping/model calls; identify backend and memory scope."""
    if steps < 1 or prediction_repeats < 1:
        raise ValueError("Resource probes must contain at least one iteration")
    snapshot = env.snapshot()
    simulation = (
        env.spawn_from_snapshot(snapshot)
        if hasattr(env, "spawn_from_snapshot")
        else env.fork(snapshot)
    )
    history = simulation.history_features()
    native_process = getattr(getattr(simulation, "_driver", None), "process", None)
    worker_pid = None if native_process is None else native_process.pid
    native_memory_before = process_tree_memory(worker_pid)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    start, advanced = time.perf_counter(), 0
    for _ in range(steps):
        if simulation.done:
            break
        simulation.advance("nominal")
        advanced += 1
    simulation_seconds = time.perf_counter() - start
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    native_memory_after = process_tree_memory(worker_pid)
    prediction_seconds = []
    gpu_memory = None
    if model is not None:
        candidates = list(
            candidates
            or [Candidate("infrastructure", "cautious", 0.5), Candidate("supervisor", "hold", 2.0)]
        )
        model.predict(history, candidates)  # warm-up reported separately
        if str(model.config.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(prediction_repeats):
            start = time.perf_counter()
            model.predict(history, candidates)
            if str(model.config.device).startswith("cuda"):
                torch.cuda.synchronize()
            prediction_seconds.append(time.perf_counter() - start)
        if str(model.config.device).startswith("cuda"):
            gpu_memory = int(torch.cuda.max_memory_allocated())
    gpu_device_memory = None
    command = shutil.which("nvidia-smi")
    if command:
        process = subprocess.run(
            [command, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if process.returncode == 0:
            gpu_device_memory = process.stdout.strip().splitlines()
    close = getattr(simulation, "close", None)
    if close is not None:
        close()
    return {
        "backend": env.backend_name,
        "backend_class": f"{type(env).__module__}.{type(env).__name__}",
        "native_simulator_profiled": env.backend_name in ("isaac", "gazebo"),
        "steps": advanced,
        "simulation_seconds": simulation_seconds,
        "simulated_seconds": advanced * env.config.dt,
        "simulation_steps_per_second": advanced / simulation_seconds
        if simulation_seconds
        else None,
        "process_peak_rss_bytes_before": rss_before,
        "process_peak_rss_bytes_after": rss_peak,
        "memory_scope": "Whole process high-water RSS; not isolated simulator allocation",
        "native_worker_memory_before": native_memory_before,
        "native_worker_memory_after": native_memory_after,
        "model_device": None if model is None else str(model.config.device),
        "model_latency_seconds_mean": float(np.mean(prediction_seconds))
        if prediction_seconds
        else None,
        "model_latency_seconds_max": float(max(prediction_seconds)) if prediction_seconds else None,
        "model_prediction_repeats": len(prediction_seconds),
        "model_warmup_calls": int(model is not None),
        "model_cuda_peak_allocated_bytes": gpu_memory,
        "gpu_device_memory_csv": gpu_device_memory,
        "gpu_device_memory_scope": "Whole-device totals include all jobs; not isolated native-simulator GPU allocation",
        "native_scene_startup_included": False,
    }
