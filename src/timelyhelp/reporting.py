"""Reproducible figures and diagnostic research reports from measured artifacts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import hierarchical_bootstrap, pareto_frontier


def _json(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=_json, allow_nan=False).encode()
    ).hexdigest()


def _save(path: Path, value: Any):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json, allow_nan=False) + "\n"
    )


def _empty(ax, reason: str):
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
    ax.set_axis_off()


def method_summary(records: list[dict]) -> list[dict]:
    metrics = (
        "success",
        "completed",
        "collision",
        "near_misses",
        "elapsed_time",
        "distance",
        "queries",
        "reply_delay",
        "communication_bytes",
        "supervisor_workload",
        "late_requests",
        "useless_requests_audited",
        "deadline_error",
    )
    result = []
    for method in sorted({record["method"] for record in records}):
        rows = [record for record in records if record["method"] == method]
        summary = {"method": method, "missions": len(rows)}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            summary[metric] = float(np.mean(values)) if values else None
        times = [
            float(row["completion_time"]) for row in rows if row.get("completion_time") is not None
        ]
        summary["completion_time_conditional"] = float(np.mean(times)) if times else None
        result.append(summary)
    return result


def frozen_gate_audit(records: list[dict]) -> dict:
    """Check recorded primary/transfer gate identity before labeling a figure."""
    primary, transfer = {}, {}
    for row in records:
        weight = row.get("gate_weight_sha256")
        if not weight:
            continue
        key = (row.get("controller", "local"), row["training_seed"], row["method"])
        identity = (weight, row.get("gate_selection_sha256"))
        if row.get("backend") in ("isaac", "isaac_dynanav"):
            destination = primary
        elif row.get("backend") == "gazebo" and row.get("frozen_gate"):
            destination = transfer
        else:
            continue
        if key in destination and destination[key] != identity:
            return {
                "verified": False,
                "status": "inconsistent gate identity within a native cohort",
            }
        destination[key] = identity
    if not primary or not transfer:
        return {
            "verified": False,
            "status": "requires recorded primary Isaac and frozen Gazebo gate identities",
        }
    if any(primary.get(key) != identity for key, identity in transfer.items()):
        return {
            "verified": False,
            "status": "transfer weights/selection do not match recorded primary gates",
        }
    return {
        "verified": True,
        "status": "all recorded transfer gate identities match primary",
        "matched_gates": len(transfer),
        "selection_provenance_complete": all(
            identity[1] is not None for identity in transfer.values()
        ),
    }


def sampled_windows(
    delays, information, collision, *, risk_limit=0.1, benefit_threshold=0.0
) -> dict:
    """Identify all supported grid intervals, retaining disconnections/censoring.

    Endpoints are sampled delays, not interpolated exact continuous deadlines.
    At the right edge a positive window is explicitly right-censored. No window
    is claimed beyond the tested horizon, or when no delay satisfies both tests.
    """
    delays, information, collision = (
        np.asarray(x, dtype=float) for x in (delays, information, collision)
    )
    if delays.ndim != 1 or not (delays.shape == information.shape == collision.shape):
        raise ValueError("Window arrays must be one dimensional and matched")
    if not np.isfinite(delays).all() or len(set(delays.tolist())) != len(delays):
        raise ValueError("Window delays must be unique and finite")
    order = np.argsort(delays)
    delays, information, collision = delays[order], information[order], collision[order]
    good = (information > benefit_threshold) & (collision <= risk_limit)
    intervals = []
    start = None
    for index, useful in enumerate(good):
        if useful and start is None:
            start = index
        if start is not None and (not useful or index == len(good) - 1):
            end = index if useful else index - 1
            intervals.append([float(delays[start]), float(delays[end])])
            start = None
    censored = bool(len(good) and good[-1])
    return {
        "intervals": intervals,
        "right_censored": censored,
        "connected_deadline": intervals[0][1] if len(intervals) == 1 and not censored else None,
        "disconnected": len(intervals) > 1,
        "resolution": "sampled delay grid",
    }


def window_diagnostics(
    rows: list[dict], predictions: list[dict] | None = None, *, risk_limit=0.1
) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row.get("layout"),
            row.get("rollout_id"),
            row["snapshot_id"],
            row["source"],
            row["bridge"],
        )
        groups[key][float(row["delay"])].append(row)
    predicted = defaultdict(lambda: defaultdict(dict))
    for prediction in predictions or []:
        key = (
            prediction.get("layout"),
            prediction.get("rollout_id"),
            prediction["snapshot_id"],
            prediction["source"],
            prediction["bridge"],
        )
        predicted[key][prediction.get("training_seed")][float(prediction["delay"])] = prediction
    result = []
    for key, delays in sorted(groups.items(), key=lambda pair: str(pair[0])):
        grid = sorted(delays)
        info = [float(np.mean([r["info"] for r in delays[delay]])) for delay in grid]
        risk = [
            float(np.mean([r["outcomes"]["Q"]["collision"] for r in delays[delay]]))
            for delay in grid
        ]
        actual = sampled_windows(grid, info, risk, risk_limit=risk_limit)
        entry = dict(zip(("layout", "rollout_id", "snapshot_id", "source", "bridge"), key))
        entry.update(
            actual=actual,
            delays=grid,
            info=info,
            risk=risk,
            replications=[len(delays[delay]) for delay in grid],
        )
        seed_reports = []
        for training_seed, prediction_rows in predicted[key].items():
            pred_grid = sorted(prediction_rows)
            seed_risk_limit = prediction_rows[pred_grid[0]].get("risk_limit", risk_limit)
            actual_seed = sampled_windows(grid, info, risk, risk_limit=seed_risk_limit)
            predicted_window = sampled_windows(
                pred_grid,
                [prediction_rows[delay]["predicted_info"] for delay in pred_grid],
                [prediction_rows[delay]["predicted_collision"] for delay in pred_grid],
                risk_limit=seed_risk_limit,
            )
            a, p = actual_seed["connected_deadline"], predicted_window["connected_deadline"]
            seed_report = {
                "training_seed": training_seed,
                "predicted": predicted_window,
                "actual_at_selected_risk_limit": actual_seed,
                "risk_limit": seed_risk_limit,
                "deadline_absolute_error": abs(a - p) if a is not None and p is not None else None,
            }
            # Compare support on observed delays only. Absence beyond the grid is not a negative label.
            common = [d for d in grid if d in prediction_rows]
            seed_report["grid_classification_error"] = (
                float(
                    np.mean(
                        [
                            (
                                (info[grid.index(d)] > 0 and risk[grid.index(d)] <= seed_risk_limit)
                                != (
                                    prediction_rows[d]["predicted_info"] > 0
                                    and prediction_rows[d]["predicted_collision"] <= seed_risk_limit
                                )
                            )
                            for d in common
                        ]
                    )
                )
                if common
                else None
            )
            seed_reports.append(seed_report)
        if seed_reports:
            entry["predictions_by_training_seed"] = seed_reports
            entry["predicted"] = seed_reports[0]["predicted"] if len(seed_reports) == 1 else None
            for metric in ("deadline_absolute_error", "grid_classification_error"):
                values = [r[metric] for r in seed_reports if r[metric] is not None]
                entry[metric] = float(np.mean(values)) if values else None
        result.append(entry)
    return result


def pilot_hypotheses(
    rows: list[dict],
    records: list[dict],
    *,
    learned_method="timelyhelp",
    strongest_comparator="model_based_voi",
    bootstrap_samples=1000,
    seed=0,
) -> dict:
    """Quantify planned gates; keep data coverage and empirical findings distinct."""
    by_curve = defaultdict(list)
    for row in rows:
        by_curve[(row.get("layout"), row["snapshot_id"], row["source"], row["bridge"])].append(row)
    sign_change, varying = 0, 0
    for curve in by_curve.values():
        delay_groups = defaultdict(list)
        for row in curve:
            delay_groups[row["delay"]].append(float(row["info"]))
        effects = np.asarray([np.mean(values) for values in delay_groups.values()])
        sign_change += bool(np.any(effects > 0) and np.any(effects < 0))
        varying += bool(len(effects) > 1 and np.ptp(effects) > 0.01)
    info = [float(row["info"]) for row in rows]
    output = {
        "status": "pilot diagnostics only; no publication or performance claim",
        "curves": len(by_curve),
        "curves_with_sign_change": sign_change,
        "curves_with_effect_range_above_0_01": varying,
        "positive_information_fraction": float(np.mean(np.asarray(info) > 0)) if info else None,
        "negative_information_fraction": float(np.mean(np.asarray(info) < 0)) if info else None,
        "neutral_information_fraction": float(np.mean(np.asarray(info) == 0)) if info else None,
        "information_survives_motion_matching": bool(any(value > 0 for value in info)),
        "oracle_gap": "not assessed unless an explicitly privileged oracle mission method was run",
        "predictability": "requires held-out window/calibration diagnostics and independent training seeds",
    }
    methods = {record["method"] for record in records}
    if learned_method in methods and strongest_comparator in methods:
        completion = hierarchical_bootstrap(
            records,
            "success",
            comparison=(learned_method, strongest_comparator),
            samples=bootstrap_samples,
            seed=seed,
        )
        requests = hierarchical_bootstrap(
            records,
            "queries",
            comparison=(learned_method, strongest_comparator),
            samples=bootstrap_samples,
            seed=seed,
        )
        output["completion_gain"] = completion
        output["request_difference"] = requests
        output["planned_completion_target_met_in_point_estimate"] = (
            completion.get("difference", {}).get("estimate", -np.inf) >= 0.05
        )
        # A request reduction target alone is insufficient; collateral outcomes stay explicit.
        summaries = {row["method"]: row for row in method_summary(records)}
        q1, q0 = summaries[learned_method]["queries"], summaries[strongest_comparator]["queries"]
        output["request_reduction_fraction"] = (
            (q0 - q1) / q0 if q0 is not None and q0 > 0 and q1 is not None else None
        )
        output["target_comparability"] = (
            "collision and completion-time equivalence margins require predeclared calibration; not inferred here"
        )
    return output


def _scene_examples(rows: list[dict]) -> list[dict]:
    """Pick outcome-different cases matched on visible-confidence summaries.

    Differences are explained using recorded public map/metadata, not actor IDs,
    corruption flags or latent truth. The matching tolerance is published.
    """
    cases = {}
    for row in rows:
        observation = row.get("observation", row.get("public_observation"))
        if observation is None:
            continue
        key = (row.get("layout"), row["snapshot_id"], row["source"], row["bridge"], row["delay"])
        if key in cases:
            continue
        tracks = observation.get("tracks", [])
        confidence = (
            1.0 / (1.0 + float(np.mean([track["covariance"] for track in tracks])))
            if tracks
            else 0.0
        )
        cases[key] = {
            "layout": row.get("layout"),
            "snapshot_id": row["snapshot_id"],
            "source": row["source"],
            "bridge": row["bridge"],
            "delay": row["delay"],
            "information_benefit": row["info"],
            "visible_track_count": len(tracks),
            "confidence_summary": confidence,
            "accessible_evidence": {
                k: observation.get(k)
                for k in (
                    "time",
                    "pose",
                    "velocity",
                    "goal",
                    "crossings",
                    "blind_zone_uncertainty",
                    "sources",
                    "remaining_time",
                )
            },
        }
    pairs = []
    values = list(cases.values())
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            if (
                first["snapshot_id"] != second["snapshot_id"]
                and first["visible_track_count"] == second["visible_track_count"]
                and abs(first["confidence_summary"] - second["confidence_summary"]) <= 0.05
                and (first["information_benefit"] > 0) != (second["information_benefit"] > 0)
            ):
                pairs.append(
                    {
                        "matching": "same visible-track count and confidence difference <= 0.05",
                        "first": first,
                        "second": second,
                    }
                )
                if len(pairs) >= 3:
                    return pairs
    return pairs


def generate_report(
    records: list[dict],
    counterfactual_rows: list[dict],
    output_dir: str | Path,
    config: dict | None = None,
    *,
    bootstrap_samples=1000,
    seed=0,
    window_predictions: list[dict] | None = None,
    scene_audit: dict | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = config or {}
    backend_records = {
        backend: [r for r in records if r.get("backend", "reference") == backend]
        for backend in sorted({r.get("backend", "reference") for r in records})
    }
    metrics = ("success", "completed", "collision", "elapsed_time", "queries")
    intervals_by_backend = {
        backend: {
            metric: hierarchical_bootstrap(rows, metric, samples=bootstrap_samples, seed=seed)
            for metric in metrics
        }
        for backend, rows in backend_records.items()
    }
    summaries_by_backend = {
        backend: method_summary(rows) for backend, rows in backend_records.items()
    }
    frontiers_by_backend = {
        backend: pareto_frontier(rows) for backend, rows in summaries_by_backend.items()
    }
    # Different native dynamics/sensors are distinct evaluation cohorts. Keep
    # their mission keys, confidence intervals and Pareto comparisons separate.
    multiple_backends = len(backend_records) > 1
    summaries, frontier = [], []
    for backend, rows in summaries_by_backend.items():
        for row in rows:
            labeled = dict(
                row,
                backend=backend,
                method=f"{row['method']} @ {backend}" if multiple_backends else row["method"],
            )
            summaries.append(labeled)
            if row in frontiers_by_backend[backend]:
                frontier.append(labeled)
    intervals = (
        next(iter(intervals_by_backend.values()))
        if len(intervals_by_backend) == 1
        else {
            metric: {"status": "distinct simulator cohorts; see confidence_intervals_by_backend"}
            for metric in metrics
        }
    )
    window_risk_limit = config.get(
        "window_risk_limit",
        config.get("calibration", {}).get("max_collision_rate", config.get("risk_limit", 0.1)),
    )
    windows = window_diagnostics(
        counterfactual_rows, window_predictions, risk_limit=window_risk_limit
    )
    hypotheses_by_backend = {
        backend: pilot_hypotheses(
            [r for r in counterfactual_rows if r.get("backend", "reference") == backend],
            rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for backend, rows in backend_records.items()
    }
    hypotheses = (
        next(iter(hypotheses_by_backend.values()))
        if len(hypotheses_by_backend) == 1
        else {"status": "distinct simulator cohorts; see pilot_by_backend"}
    )
    examples = [] if scene_audit is not None else _scene_examples(counterfactual_rows)
    gate_audit = frozen_gate_audit(records)
    previous_provenance = (
        json.loads((output / "provenance.json").read_text())
        if (output / "provenance.json").is_file()
        else {}
    )
    provenance = {
        **previous_provenance,
        "config": config,
        "config_sha256": _hash(config),
        "mission_records_sha256": _hash(records),
        "counterfactual_sha256": _hash(counterfactual_rows),
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "experiment_scale": config.get("mode", "unspecified"),
        "backend_counts": {
            backend: sum(r.get("backend", "reference") == backend for r in records)
            for backend in sorted({r.get("backend", "reference") for r in records})
        },
    }
    report = {
        "provenance": provenance,
        "summary": summaries,
        "pareto_frontier": frontier,
        "confidence_intervals": intervals,
        "windows": windows,
        "pilot": hypotheses,
        "summary_by_backend": summaries_by_backend,
        "pareto_frontier_by_backend": frontiers_by_backend,
        "confidence_intervals_by_backend": intervals_by_backend,
        "pilot_by_backend": hypotheses_by_backend,
        "window_risk_limit": window_risk_limit,
        "native_frozen_gate_audit": gate_audit,
        "accessible_scene_pairs": examples,
        "matched_observable_scene_audit": scene_audit,
        "limitations": [
            "Smoke runs do not establish effectiveness or safety guarantees.",
            "Counterfactual usefulness estimates are conditional Monte Carlo labels; few replications are noisy.",
            "A missing native backend is not replaced by a differently named reference simulation.",
        ],
    }
    _save(output / "report.json", report)
    _save(output / "provenance.json", provenance)
    _save(output / "accessible_scene_pairs.json", examples)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = defaultdict(lambda: defaultdict(list))
    for row in counterfactual_rows:
        groups[(row["source"], row["bridge"])][row["delay"]].append(row)
    for (source, bridge), grid in sorted(groups.items()):
        delays = sorted(grid)
        for axis, key in zip(axes, ("info", "motion")):
            axis.plot(
                delays,
                [np.mean([r[key] for r in grid[delay]]) for delay in delays],
                "o-",
                label=f"{source}/{bridge}",
            )
    for axis, title in zip(axes, ("Information effect W − Q", "Motion effect A − W")):
        axis.axhline(0, color="grey", linewidth=0.7)
        axis.set(xlabel="Arrival delay (s)", ylabel="Cost improvement", title=title)
        if groups:
            axis.legend(fontsize=7)
        else:
            _empty(axis, "No counterfactual labels recorded")
    files.append(_finish(fig, output, "01_delay_effects"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in summaries:
        for ax, x, y in ((axes[0], "queries", "success"), (axes[1], "elapsed_time", "collision")):
            if row[x] is not None and row[y] is not None:
                ax.scatter(row[x], row[y], marker="*" if row in frontier else "o")
                ax.annotate(row["method"], (row[x], row[y]), fontsize=7)
    axes[0].set(xlabel="Requests per mission", ylabel="Completion without collision within budget")
    axes[1].set(xlabel="Mission elapsed time including failures (s)", ylabel="Collision rate")
    if not summaries:
        for ax in axes:
            _empty(ax, "No mission results recorded")
    files.append(_finish(fig, output, "02_safety_efficiency_queries"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    errors = [
        w["deadline_absolute_error"]
        for w in windows
        if w.get("deadline_absolute_error") is not None
    ]
    classification = [
        w["grid_classification_error"]
        for w in windows
        if w.get("grid_classification_error") is not None
    ]
    if errors:
        axes[0].hist(errors, bins=min(10, max(1, len(errors))))
        axes[0].set(xlabel="Uncensored connected-window deadline error (s)", ylabel="Windows")
    elif classification:
        axes[0].hist(classification, bins=min(10, max(1, len(classification))))
        axes[0].set(xlabel="Useful-grid classification error", ylabel="Windows")
    else:
        _empty(
            axes[0],
            "No matched uncensored deadline predictions;\nwindows and censoring are in report.json",
        )
    usable = [row for row in summaries if row["late_requests"] is not None]
    if usable:
        axes[1].bar(
            np.arange(len(usable)),
            [row["late_requests"] for row in usable],
            label="Expired/stale arrivals",
        )
        audited = [row["useless_requests_audited"] for row in usable]
        if all(x is not None for x in audited):
            axes[1].bar(
                np.arange(len(usable)), audited, alpha=0.5, label="Audited nonpositive information"
            )
        axes[1].set_xticks(
            np.arange(len(usable)),
            [r["method"] for r in usable],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        axes[1].set_ylabel("Requests per mission")
        axes[1].legend(fontsize=7)
    else:
        _empty(axes[1], "No request diagnostics recorded")
    files.append(_finish(fig, output, "03_useful_windows_requests"))

    fig, ax = plt.subplots(figsize=(10, 4))
    disturbed = defaultdict(lambda: defaultdict(list))
    for record in records:
        disturbances = record.get("disturbances") or {}
        label = (
            json.dumps(disturbances, sort_keys=True)
            if isinstance(disturbances, dict)
            else str(disturbances)
        )
        method = (
            f"{record['method']} @ {record['backend']}" if multiple_backends else record["method"]
        )
        disturbed[label][method].append(float(record["success"]))
    labels = sorted(disturbed)
    for method in sorted({r["method"] for r in summaries}):
        ax.plot(
            range(len(labels)),
            [
                np.mean(disturbed[label][method]) if disturbed[label][method] else np.nan
                for label in labels
            ],
            "o-",
            label=method,
        )
    if labels:
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=6)
        ax.set(
            ylabel="Safe completion rate",
            title="Recorded disturbance composition; held-out status follows split manifest",
        )
        ax.legend(fontsize=7)
    else:
        _empty(ax, "No disturbance missions recorded")
    files.append(_finish(fig, output, "04_disturbance_composition"))

    fig, ax = plt.subplots(figsize=(10, 4))
    native = [r for r in records if r.get("backend") in ("gazebo", "isaac", "isaac_dynanav")]
    backends = sorted({r["backend"] for r in native})
    frozen_hash = config.get("frozen_gate_sha256")
    if len(backends) >= 2 and frozen_hash and gate_audit["verified"]:
        for method in sorted({r["method"] for r in native}):
            values = [
                np.mean(
                    [
                        r["success"]
                        for r in native
                        if r["backend"] == backend and r["method"] == method
                    ]
                )
                if any(r["backend"] == backend and r["method"] == method for r in native)
                else np.nan
                for backend in backends
            ]
            ax.plot(backends, values, "o-", label=method)
        ax.set(
            ylabel="Safe completion",
            title=f"Native cross-simulator test; frozen gate {frozen_hash[:12]}",
        )
        ax.legend(fontsize=7)
    else:
        _empty(
            ax,
            "Native cross-simulator results unavailable.\nRequire two native backends with matching recorded frozen gates.\n"
            + gate_audit["status"]
            + ".\nReference dynamics variants are not Gazebo transfer.",
        )
    files.append(_finish(fig, output, "05_frozen_gate_transfer"))

    fig, ax = plt.subplots(figsize=(11, 5))
    if scene_audit and scene_audit.get("identical_confidence_pairs"):
        texts = []
        for pair in scene_audit["identical_confidence_pairs"][:2]:
            differences = pair["accessible_differences"]
            details = "; ".join(
                f"{d['feature']}: {d['first_current']:.3g} → {d['second_current']:.3g}"
                for d in differences
            )
            texts.append(
                f"{pair['first_snapshot']} vs {pair['second_snapshot']}\n"
                f"Matched confidence {pair['first_confidence']:.5g}/{pair['second_confidence']:.5g}; "
                f"oracle choices {pair['first_choice']} / {pair['second_choice']}\n"
                f"Accessible evidence differences: {details}"
            )
        _empty(
            ax,
            "\n\n".join(texts)
            + "\n\nChoices are conditional-label oracle diagnostics; all feature differences recorded in matched_observable_scenes.json",
        )
    elif scene_audit and scene_audit.get("identical_observation_aliases"):
        _empty(
            ax,
            "Identical observable histories had different sampled oracle choices.\n"
            "These are observational aliases: deployable evidence cannot reliably distinguish them.\n"
            "All cases recorded in matched_observable_scenes.json.",
        )
    elif examples:
        texts = []
        for pair in examples:
            for label in ("first", "second"):
                case = pair[label]
                evidence = case["accessible_evidence"]
                texts.append(
                    f"{case['snapshot_id']}  confidence={case['confidence_summary']:.3f}  "
                    f"source={case['source']} delay={case['delay']} bridge={case['bridge']}\n"
                    f"observed information effect={case['information_benefit']:.3f}; "
                    f"robot pose={evidence.get('pose')}; velocity={evidence.get('velocity')}; "
                    f"blind-zone uncertainty={evidence.get('blind_zone_uncertainty')}"
                )
        _empty(
            ax,
            "\n\n".join(texts)
            + "\n\nFull public evidence and matching tolerances: accessible_scene_pairs.json",
        )
    else:
        _empty(
            ax,
            "No outcome-different pairs matched the declared visible-confidence tolerance.\n"
            "Collect annotated public observation examples; no pairs are fabricated.",
        )
    files.append(_finish(fig, output, "06_accessible_evidence_scenes"))

    lines = [
        "# TimelyHelp measured artifact report",
        "",
        "This is a smoke/pilot report, not a full-scale experiment.",
        "",
        f"Mission records: {len(records)}; counterfactual label rows: {len(counterfactual_rows)}.",
        "",
        "All raw outcomes, matched hierarchical intervals, disconnected/right-censored useful windows, "
        "and provenance hashes are in `report.json`.",
        "",
        "The primary outcome is completion without collision within the mission budget. "
        "Failures retain their elapsed time; conditional completion time is reported separately.",
        "",
        "Bootstrap units are independent training seeds, layouts, and complete matched mission rollouts. "
        "Counterfactual siblings are excluded. Five independent fitted seeds and sufficient layouts are "
        "required before assessing the proposed confirmatory protocol.",
        "",
        "No available native cross-simulator evidence is inferred from a reference simulator.",
        "",
        "Figures:",
        "",
    ]
    lines += [f"- [{Path(file).name}]({Path(file).name})" for file in files]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return {
        "report": str(output / "report.json"),
        "markdown": str(output / "REPORT.md"),
        "figures": files,
    }


def _finish(fig, output: Path, name: str) -> str:
    fig.tight_layout()
    path = output / f"{name}.png"
    fig.savefig(path, dpi=150)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)
    return str(path)
