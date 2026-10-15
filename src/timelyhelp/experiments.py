"""Complete bounded/research stage orchestration with immutable measured artifacts.

The configuration declares smoke, pilot or confirmatory scale. Every split is a
whole rollout in a disjoint physical layout; candidate siblings never cross it.
Full-scale commands exist but are never invoked by an implementation smoke run.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta

from .baselines import BaselineConfig, DiagnosticOracle, calibrate_source_metadata, make_baseline
from .config import digest, provenance, write_json
from .counterfactual import CounterfactualRunner
from .evaluation import run_episode, run_evaluation, hierarchical_bootstrap, pareto_frontier
from .learning.data import PairedDataset, content_hash
from .learning.model import ConsequenceEnsemble, ModelConfig
from .learning.scheduler import LearnedScheduler, SchedulerConfig
from .reporting import generate_report, sampled_windows, method_summary
from .simulation.reference import ReferenceSimulator
from .sim_pool import (
    ACCEPTED_GENERATION_CODES,
    code_hash,
    configured_gpus,
    execute_jobs,
    parallel_simulations,
    read_valid_shard,
    status_directory,
    touch_status,
    work_hash,
)
from .simulation.scenarios import (
    DISTURBANCES,
    FAMILIES,
    WITHHELD_COMPOSITIONS,
    Scenario,
    make_scenario,
)
from .types import Candidate, SimConfig
from .audits import (
    audit_public_observation,
    audit_controlled_labels,
    pilot_diagnostics,
    monte_carlo_summary,
    matched_observable_scenes,
    profile_resources,
    window_prediction_grid,
)


def _read(path):
    return json.loads(Path(path).read_text())


def _stage(output, config, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "provenance.json", provenance(config, stage=name))
    return output


def _scenario_list(config, split, count):
    """Balance event family and mobility setting before increasing layout variants."""
    if count < 1:
        raise ValueError("Every requested stage must include at least one complete episode")
    disturbances = [()] + [(d,) for d in DISTURBANCES]
    if split == "test":
        disturbances += list(WITHHELD_COMPOSITIONS)
    offset = {"train": 0, "development": 2, "calibration": 4, "test": 6}[split]
    result = []
    for index in range(count):
        # Smoke counts intentionally sample a bounded subset of the full suite.
        disturbance = disturbances[(index + offset) % len(disturbances)]
        if split == "test" and config["mode"] == "smoke" and index == count - 1 and count > 1:
            disturbance = WITHHELD_COMPOSITIONS[0]
        family = FAMILIES[index % len(FAMILIES)]
        setting = ("worksite", "sidewalk")[index % 2]
        variant = index // (6 * len(disturbances))
        result.append(make_scenario(split, setting, family, variant, disturbance))
    return result


def _transfer_scenarios(config, count):
    """Three physical corridor/junction layouts, with balanced behavior probes.

    The mobility setting is held fixed here: using both settings would silently
    double the declared three-layout transfer benchmark. Short smoke counts may
    cover only a subset, which the evaluation manifest reports explicitly.
    """
    scenarios = _scenario_list(config, "test", count)
    return [
        make_scenario("test", "worksite", scenario.family, index % 3, scenario.disturbances)
        for index, scenario in enumerate(scenarios)
    ]


def _environment(config, scenario, seed, controller="local", source_calibration=None):
    simulation = SimConfig.from_dict(config.get("simulation", {}) | {"controller": controller})
    if config["backend"] == "reference":
        env = ReferenceSimulator(scenario, seed, simulation)
    else:
        from .backends.native import backend_factory

        env = backend_factory(config["backend"], scenario, seed, simulation)
    for source, metadata in (source_calibration or {}).items():
        stats = env.source_stats[source]
        stats["reliability_prior"] = metadata["reliability_prior"]
        stats["reliability_calibration_count"] = metadata["calibration_verifications"]
        stats["reliability_calibration"] = metadata["calibration_status"]
        stats["response_prior_alpha"] = metadata.get("response_alpha", 1.0)
        stats["response_prior_beta"] = metadata.get("response_beta", 1.0)
    # Frozen public metadata are included in the initial history too.
    if source_calibration:
        env.history = []
        env._record_history()
    return env


def _close(env):
    method = getattr(env, "close", None)
    if method is not None:
        method()


def _source_audits(env, snapshot):
    """Privileged *development-only* packet quality labels, stored separately."""
    if snapshot["scenario"].split != "development":
        raise ValueError("Source audit cannot use another split")
    rows = []
    for source in ("infrastructure", "supervisor"):
        fork = env.fork(snapshot)
        request_id = fork.request(Candidate(source, "hold", 0.0), exact_delay=0.0)
        message = next(m for m in fork.pending if m.request_id == request_id)
        response = np.isfinite(message.arrival_time)
        payload = message.payload
        correct = None
        if response and payload.get("answer") != "unknown":
            if source == "supervisor":
                # The bounded worker source can know only its own intended route.
                correct = (payload["answer"] == "cross") == bool(fork.actors[0]["crosses"])
            else:
                # Source truth audits never replace sensor-generated payloads.
                matched = []
                for detection in payload.get("tracks", []):
                    candidates = [
                        actor for actor in fork.actors if actor["kind"] == detection["kind"]
                    ]
                    matched.append(
                        bool(candidates)
                        and min(
                            np.linalg.norm(np.asarray(detection["position"]) - actor["position"])
                            for actor in candidates
                        )
                        <= 0.5
                    )
                correct = bool(matched and all(matched))
        rows.append(
            {
                "split": "development",
                "source": source,
                "response": bool(response),
                "correct": correct,
                "capture_time": message.capture_time,
                "layout": fork.scenario.layout_id,
                "seed": fork.seed,
                "audit_privilege": "offline actor-state correctness labels; never policy input",
            }
        )
        _close(fork)
    return rows


def plan_generation_jobs(config):
    """One job is one complete episode. Sibling A/W/Q branches stay on that job."""
    generation = config["generation"]
    jobs = []
    for controller in config.get("controllers", ["local"]):
        for split_index, split in enumerate(("train", "development", "calibration", "test")):
            scenarios = _scenario_list(config, split, int(generation["episodes"][split]))
            for episode_index, scenario in enumerate(scenarios):
                seed = int(config.get("scenario_seed", 0)) + 10000 * split_index + episode_index
                jobs.append(
                    {
                        "id": f"generate/{controller}/{split}/{episode_index:04d}",
                        "controller": controller,
                        "split": split,
                        "split_index": split_index,
                        "episode_index": episode_index,
                        "seed": seed,
                        "scenario": scenario.to_dict(),
                    }
                )
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Generation schedule duplicated an episode")
    return jobs


def _generation_shard(output, job):
    return (
        Path(output)
        / "shards"
        / job["controller"]
        / job["split"]
        / f"{job['episode_index']:04d}.json"
    )


def _prepare_generation_job(config, output, job, audit_restoration):
    prepared = {
        **job,
        "kind": "generate_episode",
        "audit_restoration": bool(audit_restoration),
        "config": config,
        "status_dir": str(status_directory(output)),
        "shard": str(_generation_shard(output, job)),
    }
    identity = {
        "id": prepared["id"],
        "controller": prepared["controller"],
        "split": prepared["split"],
        "episode_index": prepared["episode_index"],
        "seed": prepared["seed"],
        "scenario": prepared["scenario"],
        "audit_restoration": prepared["audit_restoration"],
    }
    prepared["work_sha256"] = work_hash("generation", config, identity)
    prepared["legacy_work_hashes"] = [
        work_hash("generation", config, identity, code=code) for code in ACCEPTED_GENERATION_CODES
    ]
    return prepared


def _collect_episode(config, controller, scenario, seed, *, audit_restoration, progress=None):
    """Prefix, snapshots, and matched continuations for one seeded rollout."""
    generation = config["generation"]
    env = _environment(config, scenario, seed, controller)
    runner = CounterfactualRunner(env)
    records, audits, restoration, resources = [], [], [], []
    try:
        times = generation.get("snapshot_times", [0.0])
        times = sorted(
            float(t) for t in times[: int(generation.get("snapshots_per_episode", len(times)))]
        )
        if len(times) != int(generation.get("snapshots_per_episode", len(times))):
            raise ValueError("snapshot_times must provide every requested snapshot")
        for snapshot_time in times:
            while not env.done and env.time < snapshot_time - 1e-9:
                env.advance("nominal")
            if env.done:
                break
            snapshot = env.snapshot()
            if audit_restoration and not restoration:
                if progress:
                    progress(f"restoration audit at t={snapshot_time:.2f}s")
                audit = runner.audit_restore(
                    snapshot,
                    steps=min(3, int(np.ceil((env.config.horizon - env.time) / env.config.dt))),
                )
                audit["controller"] = controller
                restoration.append(audit)
                if not audit["passed"]:
                    raise AssertionError(
                        "Backend restoration failed; counterfactual labels are invalid"
                    )
                resources.append(profile_resources(env, steps=2) | {"controller": controller})
            if scenario.split == "development":
                audits.extend(_source_audits(env, snapshot))
            candidates = list(
                itertools.product(
                    generation.get("sources", ["infrastructure", "supervisor"]),
                    generation.get("bridges", ["cautious", "hold"]),
                    generation.get("delays", [0.0, 0.25, 0.75, 1.5, 3.0]),
                )
            )
            replications = int(generation.get("replications", 1))
            total = len(candidates) * replications
            completed = 0
            for source, bridge, delay in candidates:
                for replication in range(replications):
                    completed += 1
                    if progress:
                        progress(
                            f"snapshot t={snapshot_time:.2f}s branch {completed}/{total} "
                            f"{source}/{bridge}/delay={float(delay):g}/rep={replication}"
                        )
                    candidate = Candidate(source, bridge, float(delay))
                    # An audit failure invalidates this job. Retrying until a
                    # pair matches hides a broken sensor/restore contract and
                    # can retain inconsistent shared A/W labels.
                    row = runner.run(snapshot, candidate, replication)
                    row["controller"] = controller
                    records.append(row)
    finally:
        _close(env)
    return {
        "records": records,
        "audits": audits,
        "restoration": restoration,
        "resources": resources,
    }


def _execute_generation(config, output, jobs):
    """Run every episode once. Only the first episode of a controller audits restoration.

    Later episodes do not depend on that audit, so they share the GPU pool with it.
    A controller whose first rollout ends before any snapshot retries the audit on
    the next episode instead of dropping the check.
    """
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["controller"], []).append(job)
    prepared = {}
    for group in grouped.values():
        for index, job in enumerate(group):
            prepared[job["id"]] = _prepare_generation_job(config, output, job, index == 0)
        if not group:
            continue
        first = prepared[group[0]["id"]]
        cached = read_valid_shard(first)
        if cached is not None and not cached.get("restoration"):
            prepared[group[0]["id"]] = _prepare_generation_job(config, output, group[0], False)
            for later in group[1:]:
                candidate = _prepare_generation_job(config, output, later, True)
                later_cached = read_valid_shard(candidate)
                prepared[later["id"]] = candidate
                if later_cached is None or later_cached.get("restoration"):
                    break
    ordered = [prepared[job["id"]] for job in jobs]
    results = {job["id"]: result for job, result in zip(ordered, execute_jobs(config, ordered))}
    for group in grouped.values():
        if any(results[job["id"]].get("restoration") for job in group):
            continue
        for job in group:
            if prepared[job["id"]]["audit_restoration"]:
                continue
            redo = _prepare_generation_job(config, output, job, True)
            Path(redo["shard"]).unlink(missing_ok=True)
            results[job["id"]] = execute_jobs(config, [redo])[0]
            prepared[job["id"]] = redo
            if results[job["id"]].get("restoration"):
                break
    missing = [job["id"] for job in jobs if job["id"] not in results]
    if missing:
        raise RuntimeError(f"Generation did not return every episode: {missing[:8]}")
    return [results[job["id"]] for job in jobs]


def _finalize_generation(config, output, jobs, results, elapsed):
    controllers = config.get("controllers", ["local"])
    controller_paths, combined_rows, audits, restoration, resources = {}, [], [], [], []
    for controller in controllers:
        records, controller_audits, scenarios_manifest = [], [], []
        for job, result in zip(jobs, results):
            if job["controller"] != controller:
                continue
            records.extend(result["records"])
            controller_audits.extend(result["audits"])
            restoration.extend(result["restoration"])
            resources.extend(result["resources"])
            scenarios_manifest.append(job["scenario"])
        if not records:
            raise ValueError("No nonterminal snapshots were collected")
        calibration = calibrate_source_metadata(controller_audits)
        for source in calibration:
            source_rows = [row for row in controller_audits if row["source"] == source]
            calibration[source]["response_alpha"] = 1 + sum(row["response"] for row in source_rows)
            calibration[source]["response_beta"] = 1 + sum(
                not row["response"] for row in source_rows
            )
        # Retrofit a frozen, observable development-calibrated prior into cached
        # features. Source metadata are not test-episode hidden parameters.
        for row in records:
            history = np.asarray(row["history"], dtype=np.float32)
            for source, index in (("infrastructure", 30), ("supervisor", 31)):
                history[:, index] = calibration[source]["reliability_prior"]
                row["observation"]["sources"][source]["reliability_prior"] = calibration[source][
                    "reliability_prior"
                ]
                row["observation"]["sources"][source]["reliability_calibration"] = calibration[
                    source
                ]["calibration_status"]
            for source, index in (("infrastructure", 22), ("supervisor", 23)):
                history[:, index] = calibration[source]["response_fraction"]
                row["observation"]["sources"][source]["response_fraction"] = calibration[source][
                    "response_fraction"
                ]
            row["history"] = history.tolist()
            audit_public_observation(row["observation"], history)
        controlled_audit = audit_controlled_labels(records)
        write_json(output / f"controlled_audit_{controller}.json", controlled_audit)
        write_json(output / f"monte_carlo_{controller}.json", monte_carlo_summary(records))
        write_json(
            output / f"pilot_{controller}.json",
            pilot_diagnostics(
                records, query_penalty=config.get("simulation", {}).get("query_cost", 0.03)
            ),
        )
        write_json(output / f"matched_scenes_{controller}.json", matched_observable_scenes(records))
        dataset = PairedDataset.from_records(records)
        dataset.provenance.update(
            controller=controller,
            backend=config["backend"],
            source_calibration=calibration,
            restoration_audits=restoration,
            data_scale=config["mode"],
            source_audit_privilege="offline development labels only",
        )
        dataset.provenance["interaction_count"] = controlled_audit["distinct_branch_interactions"]
        dataset_path = output / f"paired_{controller}.npz"
        records_path = output / f"records_{controller}.json"
        dataset.save(dataset_path)
        write_json(records_path, records)
        write_json(output / f"source_calibration_{controller}.json", calibration)
        controller_paths[controller] = {
            "dataset": str(dataset_path),
            "records": str(records_path),
            "source_calibration": str(output / f"source_calibration_{controller}.json"),
        }
        audits.extend(dict(row, controller=controller) for row in controller_audits)
        combined_rows.extend(records)
        write_json(output / f"scenarios_{controller}.json", scenarios_manifest)
    write_json(output / "privileged_development_source_audits.json", audits)
    write_json(output / "restoration_audits.json", restoration)
    write_json(output / "resource_profiles.json", resources)
    write_json(output / "records.json", combined_rows)
    main = controller_paths[controllers[0]]
    counts = {
        split: sum(row["split"] == split for row in combined_rows)
        for split in ("train", "development", "calibration", "test")
    }
    manifest = {
        "dataset": main["dataset"],
        "records": str(output / "records.json"),
        "controllers": controller_paths,
        "record_counts": counts,
        "snapshots": len({(r["controller"], r["layout"], r["snapshot_id"]) for r in combined_rows}),
        "raw_rows": len(combined_rows),
        "elapsed_seconds": elapsed,
        "filtering": "none; harmful, neutral and useful labels retained",
        "controls_shared": "A cached per snapshot/replication; W cached per bridge/delay/replication",
        "configured_disturbances": list(DISTURBANCES),
        "withheld_compositions": list(WITHHELD_COMPOSITIONS),
        "observed_disturbances": sorted({d for row in combined_rows for d in row["disturbances"]}),
        "coverage_status": "bounded smoke subset"
        if config["mode"] == "smoke"
        else "see scenario manifests for actual coverage",
        "config_sha256": digest(config),
        "code_sha256": code_hash("generation"),
        "parallel_gpus": list(configured_gpus()) if parallel_simulations(config) else [],
        "scheduling": "one seeded episode per job; A/W/Q branches are not split across GPUs",
        "generation_shards": [
            {
                "job_id": job["id"],
                "path": str(_generation_shard(output, job)),
                "sha256": hashlib.sha256(_generation_shard(output, job).read_bytes()).hexdigest(),
                "work_sha256": result["work_sha256"],
            }
            for job, result in zip(jobs, results)
        ],
        "accepted_prior_generation_codes": list(ACCEPTED_GENERATION_CODES),
        "manifest": str(output / "manifest.json"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def generate_data(config: dict, output) -> dict:
    output = _stage(output, config, "generate_data")
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = _read(manifest_path)
        if existing.get("config_sha256") == digest(config) and existing.get(
            "code_sha256"
        ) == code_hash("generation"):
            print("Generation manifest is complete; not repeating episodes.", flush=True)
            return existing
        raise ValueError(
            f"{manifest_path} belongs to a different configuration or simulator revision; "
            "use a fresh output directory"
        )
    jobs = plan_generation_jobs(config)
    print(
        f"Generating {len(jobs)} episodes on backend {config['backend']}"
        + (
            f" across GPUs {list(configured_gpus())}"
            if parallel_simulations(config)
            else " in-process"
        ),
        flush=True,
    )
    started = time.perf_counter()
    results = _execute_generation(config, output, jobs)
    return _finalize_generation(config, output, jobs, results, time.perf_counter() - started)


def _dataset_inputs(dataset):
    if isinstance(dataset, PairedDataset):
        controller = dataset.provenance.get("controller", "local")
        return {controller: (dataset, dataset.provenance.get("source_calibration", {}))}
    if isinstance(dataset, dict):
        manifest = dataset
    elif Path(dataset).suffix == ".json":
        manifest = _read(dataset)
    elif Path(dataset).is_dir():
        manifest = _read(Path(dataset) / "manifest.json")
    else:
        data = PairedDataset.load(dataset)
        return _dataset_inputs(data)
    return {
        controller: (PairedDataset.load(paths["dataset"]), _read(paths["source_calibration"]))
        for controller, paths in manifest["controllers"].items()
    }


def _unpaired(data: PairedDataset, seed: int) -> PairedDataset:
    """Independent W and Q donor shuffling, matched original interaction budget."""
    result = data.subset(range(len(data)))
    rng = np.random.default_rng(seed)
    buckets = {}
    for index, (candidate, metadata) in enumerate(zip(data.candidates, data.metadata)):
        buckets.setdefault(
            (metadata["split"], candidate.source, candidate.bridge, candidate.delay), []
        ).append(index)
    for ids in buckets.values():
        w_donors, q_donors = rng.permutation(ids), rng.permutation(ids)
        for dst, w_src, q_src in zip(ids, w_donors, q_donors):
            a = data.metadata[dst]["mean_costs"][0]
            w = data.metadata[w_src]["mean_costs"][1]
            q = data.metadata[q_src]["mean_costs"][2]
            result.targets[dst, [3, 6, 9]] = data.targets[w_src, [3, 6, 9]]
            if data.candidates[dst].source == "none":
                q = w
                result.targets[dst, [4, 7, 10]] = result.targets[dst, [3, 6, 9]]
            else:
                result.targets[dst, [4, 7, 10]] = data.targets[q_src, [4, 7, 10]]
            result.targets[dst, 0:2] = [w - q, a - w]
            result.metadata[dst] = dict(result.metadata[dst], mean_costs=[a, w, q])
    result.provenance.update(
        ablation="unpaired_W_Q_donors",
        interaction_budget="identical original simulator records",
        shuffle_seed=seed,
    )
    return result


def _threshold_calibration(
    config,
    model,
    controller,
    source_calibration,
    *,
    no_uncertainty=False,
    no_hold=False,
    model_path,
    stage_output,
    training_seed,
    variant,
):
    del model  # Workers reload the saved checkpoint so every GPU sees the same weights.
    calibration = config.get("calibration", {})
    count = int(config["generation"]["episodes"]["calibration"])
    scenarios = _scenario_list(config, "calibration", count)
    # The residual calibrator used generated calibration rollouts (+20000).
    # Selected-policy risk must be measured on new rollouts of those held-out
    # layouts, rather than recycling labels used to calibrate this model.
    policy_rollout_seeds = [
        int(config.get("scenario_seed", 0)) + 40000 + index for index in range(count)
    ]
    rows = []
    threshold_grid = list(
        itertools.product(
            calibration.get("risk_limits", [0.1]), calibration.get("query_penalties", [0.03])
        )
    )
    jobs = []
    schedulers = {}
    model_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    for risk_limit, query_penalty in threshold_grid:
        scheduler_config = SchedulerConfig(
            risk_limit=float(risk_limit),
            query_cost_weight=float(query_penalty),
            source_costs={"infrastructure": 1.0, "supervisor": 1.0},
            commitment_horizon=config.get("simulation", {}).get("commitment", 0.5),
            max_capture_age=config.get("simulation", {}).get("max_reply_age", 2.0),
            expiration=config.get("simulation", {}).get("request_expiry", 6.0),
            no_uncertainty=no_uncertainty,
            no_hold=no_hold,
        )
        schedulers[(float(risk_limit), float(query_penalty))] = scheduler_config
        for index, scenario in enumerate(scenarios):
            job_id = (
                f"calibration/{controller}/{training_seed}/{variant}/"
                f"{float(risk_limit):g}/{float(query_penalty):g}/{index}"
            )
            job = {
                "kind": "calibration_episode",
                "id": job_id,
                "config": config,
                "controller": controller,
                "training_seed": int(training_seed),
                "variant": variant,
                "seed": policy_rollout_seeds[index],
                "episode_index": index,
                "scenario": scenario.to_dict(),
                "source_calibration": source_calibration,
                "model": str(model_path),
                "scheduler": asdict(scheduler_config),
                "status_dir": str(status_directory(stage_output)),
                "shard": str(
                    Path(stage_output)
                    / "shards"
                    / "calibration"
                    / controller
                    / f"seed_{training_seed}"
                    / variant
                    / f"{float(risk_limit):g}_{float(query_penalty):g}_{index:04d}.json"
                ),
            }
            job["work_sha256"] = work_hash(
                "calibration",
                config,
                {
                    "id": job_id,
                    "seed": job["seed"],
                    "scenario": job["scenario"],
                    "scheduler": job["scheduler"],
                    "model_sha256": model_sha,
                    "source_calibration": source_calibration,
                },
            )
            jobs.append(job)
    completed = {job["id"]: result for job, result in zip(jobs, execute_jobs(config, jobs))}
    for risk_limit, query_penalty in threshold_grid:
        scheduler_config = schedulers[(float(risk_limit), float(query_penalty))]
        outcomes = [
            completed[
                f"calibration/{controller}/{training_seed}/{variant}/"
                f"{float(risk_limit):g}/{float(query_penalty):g}/{index}"
            ]["outcome"]
            for index in range(count)
        ]
        collisions = sum(outcome["collision"] for outcome in outcomes)
        upper = (
            1.0
            if collisions == len(outcomes)
            else float(beta.ppf(0.95, collisions + 1, len(outcomes) - collisions))
        )
        selected_upper = (
            1.0
            if collisions == len(outcomes)
            else float(
                beta.ppf(
                    1.0 - 0.05 / len(threshold_grid), collisions + 1, len(outcomes) - collisions
                )
            )
        )
        rows.append(
            {
                "risk_limit": float(risk_limit),
                "query_penalty": float(query_penalty),
                "collision_rate": collisions / len(outcomes),
                "collision_upper_95": upper,
                "collision_upper_familywise_95": selected_upper,
                "success": float(np.mean([o["success"] for o in outcomes])),
                "queries": float(np.mean([o["queries"] for o in outcomes])),
                "elapsed_time": float(np.mean([o["elapsed_time"] for o in outcomes])),
                "configuration": asdict(scheduler_config),
            }
        )
    feasible = [
        row
        for row in rows
        if row["collision_upper_familywise_95"] <= calibration.get("max_collision_rate", 0.1)
    ]
    chosen = (
        max(feasible, key=lambda row: (row["success"], -row["elapsed_time"], -row["queries"]))
        if feasible
        else min(rows, key=lambda row: (row["collision_upper_95"], -row["success"], row["queries"]))
    )
    report = {
        "split": "calibration",
        "evaluation": "closed_loop",
        "episodes": count,
        "rollout_seed_namespace": "scenario_seed + 40000 + calibration_episode_index",
        "rollout_seeds": policy_rollout_seeds,
        "rollout_ids": [
            f"{scenario.layout_id}:{seed}"
            for scenario, seed in zip(scenarios, policy_rollout_seeds)
        ],
        "residual_calibration_separation": "same independently held-out calibration layouts; disjoint complete rollout seeds from residual labels (+20000)",
        "thresholds": {
            "risk_limit": chosen["risk_limit"],
            "uncertainty_weight": 0.0 if no_uncertainty else 1.0,
        },
        "selected_configuration": chosen["configuration"],
        "grid_results": rows,
        "calibration_feasible": bool(feasible),
        "actual_collision_rate": chosen["collision_rate"],
        "actual_collision_upper_95": chosen["collision_upper_95"],
        "selected_collision_upper_familywise_95": chosen["collision_upper_familywise_95"],
        "threshold_selection_correction": "Bonferroni over the predeclared threshold grid; no distribution-shift guarantee",
        "warning": "Independent residual and selected-policy calibration are empirical; no distribution-shift safety guarantee.",
        "adaptation": "If no threshold meets the finite-sample collision criterion, record the least-risk observed fallback without claiming feasibility.",
    }
    return report


_MODEL_CACHE = {}


def _cached_model(path):
    filename = str(Path(path).resolve())
    key = (filename, hashlib.sha256(Path(filename).read_bytes()).hexdigest())
    if key not in _MODEL_CACHE:
        # A persistent simulation worker may see a new checkpoint at the same
        # path after a restart. Its cached network must follow the actual bytes.
        for previous in list(_MODEL_CACHE):
            if previous[0] == filename:
                del _MODEL_CACHE[previous]
        _MODEL_CACHE[key] = ConsequenceEnsemble.load(filename)
    return _MODEL_CACHE[key]


def _dataset_identity(dataset):
    if isinstance(dataset, PairedDataset):
        return content_hash(
            {
                "histories": dataset.histories,
                "candidates": [asdict(candidate) for candidate in dataset.candidates],
                "targets": dataset.targets,
                "monte_carlo_std": dataset.monte_carlo_std,
                "metadata": dataset.metadata,
                "provenance": dataset.provenance,
            }
        )
    if isinstance(dataset, dict):
        manifest = dataset
    elif Path(dataset).suffix == ".json":
        manifest = _read(dataset)
    elif Path(dataset).is_dir():
        manifest = _read(Path(dataset) / "manifest.json")
    else:
        return hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    files = [paths["dataset"] for paths in manifest["controllers"].values()]
    return digest(
        {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(files)}
    )


def _stamp_payload(config, dataset_identity, variant):
    return {
        "config_sha256": digest(config),
        "dataset_sha256": dataset_identity,
        "code_sha256": code_hash("mission"),
        "variant": variant,
    }


def _fit_identity(config, dataset_identity, variant, controller, seed, source_calibration):
    """Identity of a fit, before its independent deployed-policy calibration."""
    return {
        **_stamp_payload(config, dataset_identity, variant),
        "fit_format_version": 1,
        "controller": controller,
        "training_seed": int(seed),
        "source_calibration_sha256": digest(source_calibration),
    }


def _fit_stamp_payload(path, identity):
    path = Path(path)
    return {
        "identity": identity,
        "weight_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fit_report_sha256": hashlib.sha256(path.with_suffix(".fit.json").read_bytes()).hexdigest(),
    }


def _write_fit_stamp(path, identity):
    """Commit fit provenance only after weights and residual calibration exist."""
    write_json(Path(path).with_suffix(".fit.stamp.json"), _fit_stamp_payload(path, identity))


def _validated_fit_checkpoint(path, identity):
    """Resume only a complete, byte-verified fit with the same scientific inputs.

    Unstamped or incompatible artifacts are preserved for inspection. Silently
    recalibrating them would mislabel old weights as a fit on current data/code.
    """
    path = Path(path)
    stamp = path.with_suffix(".fit.stamp.json")
    required = (path, path.with_suffix(".fit.json"), stamp)
    if not any(item.exists() for item in required):
        return False
    message = (
        f"Unverified or incompatible fitted checkpoint at {path}; weights, fit report, "
        "and matching fit identity/checksums are required to resume. Existing artifacts "
        "were preserved; use a fresh model output directory for a new fit."
    )
    if not all(item.is_file() for item in required):
        raise ValueError(message)
    try:
        saved = _read(stamp)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(message) from error
    if saved != _fit_stamp_payload(path, identity):
        raise ValueError(message)
    return True


def _variant_stamp_matches(path, config, dataset_identity, variant):
    path = Path(path)
    required = (
        path,
        path.with_suffix(".stamp.json"),
        path.with_suffix(".selection.json"),
        path.with_suffix(".profile.json"),
        path.with_suffix(".fit.json"),
    )
    if not all(item.is_file() for item in required):
        return False
    return _read(path.with_suffix(".stamp.json")) == _stamp_payload(
        config, dataset_identity, variant
    )


def _write_variant_stamp(path, config, dataset_identity, variant):
    write_json(
        Path(path).with_suffix(".stamp.json"), _stamp_payload(config, dataset_identity, variant)
    )


def _selection_stamp_matches(stamp_path, config, dataset_identity, variant):
    stamp_path = Path(stamp_path)
    if not stamp_path.is_file():
        return False
    return _read(stamp_path) == _stamp_payload(config, dataset_identity, variant)


def _write_selection_stamp(stamp_path, config, dataset_identity, variant):
    write_json(stamp_path, _stamp_payload(config, dataset_identity, variant))


def _checkpoint_record(path):
    path = Path(path)
    selection = path.with_suffix(".selection.json")
    return {
        "model": str(path),
        "selection": str(selection),
        "weight_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }


def _write_profile(
    config, stage_output, *, model_path, controller, source_calibration, training_seed, variant
):
    scenario = _scenario_list(config, "calibration", 1)[0]
    seed = int(config.get("scenario_seed", 0)) + 20000
    model_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    job = {
        "kind": "profile",
        "id": f"profile/{controller}/{training_seed}/{variant}",
        "config": config,
        "controller": controller,
        "training_seed": int(training_seed),
        "variant": variant,
        "seed": seed,
        "scenario": scenario.to_dict(),
        "source_calibration": source_calibration,
        "model": str(model_path),
        "status_dir": str(status_directory(stage_output)),
        "shard": str(
            Path(stage_output)
            / "shards"
            / "profile"
            / controller
            / f"seed_{training_seed}"
            / f"{variant}.json"
        ),
    }
    job["work_sha256"] = work_hash(
        "calibration",
        config,
        {"id": job["id"], "seed": seed, "scenario": job["scenario"], "model_sha256": model_sha},
    )
    profile = execute_jobs(config, [job])[0]["profile"]
    write_json(Path(model_path).with_suffix(".profile.json"), profile)
    return profile


def _generate_episode_job(job):
    scenario = Scenario.from_dict(job["scenario"])

    def progress(message):
        touch_status(job, message)

    touch_status(job, "opening simulator")
    return _collect_episode(
        job["config"],
        job["controller"],
        scenario,
        job["seed"],
        audit_restoration=job["audit_restoration"],
        progress=progress,
    )


def _calibration_episode_job(job):
    touch_status(job, "calibration episode")
    model = _cached_model(job["model"])
    scheduler = LearnedScheduler(model, SchedulerConfig(**job["scheduler"]))
    env = _environment(
        job["config"],
        Scenario.from_dict(job["scenario"]),
        job["seed"],
        job["controller"],
        job["source_calibration"],
    )
    try:
        outcome = run_episode(env, scheduler)
    finally:
        _close(env)
    return {
        "outcome": {
            "collision": bool(outcome["collision"]),
            "success": bool(outcome["success"]),
            "queries": float(outcome["queries"]),
            "elapsed_time": float(outcome["elapsed_time"]),
        }
    }


def _profile_job(job):
    touch_status(job, "resource profile")
    model = _cached_model(job["model"])
    env = _environment(
        job["config"],
        Scenario.from_dict(job["scenario"]),
        job["seed"],
        job["controller"],
        job["source_calibration"],
    )
    try:
        profile = profile_resources(env, model=model, steps=2, prediction_repeats=3)
    finally:
        _close(env)
    return {"profile": profile}


def _policy_from_job(job):
    baseline = BaselineConfig.from_dict(job["baseline"])
    spec = job["policy"]
    if spec["role"] == "learned":
        model, threshold = _verified_gate(spec)
        scheduler = LearnedScheduler(model, SchedulerConfig(**threshold["selected_configuration"]))
        scheduler.configure_thresholds(threshold)
        if spec.get("query_cost_override") is not None:
            updated = asdict(scheduler.config) | {
                "query_cost_weight": float(spec["query_cost_override"])
            }
            scheduler = LearnedScheduler(model, SchedulerConfig(**updated))
        return scheduler
    if spec["role"] == "failure":
        model, _threshold = _verified_gate(spec)
        return make_baseline(
            "failure_lead_time",
            baseline,
            seed=job["training_seed"],
            failure_predictor=model,
            failure_predictor_provenance=spec["predictor_provenance"],
        )
    return make_baseline(spec["method"], baseline, seed=job["training_seed"])


def _mission_job(job):
    touch_status(job, f"mission {job['method']}")
    policy = _policy_from_job(job)
    scenario = Scenario.from_dict(job["scenario"])

    def factory(factory_scenario, rollout_seed):
        return _environment(
            job["config_for_env"],
            factory_scenario,
            rollout_seed,
            job["controller"],
            job["source_calibration"],
        )

    rows = run_evaluation(
        factory,
        {job["method"]: policy},
        [scenario],
        seeds=[int(job["rollout_seed"])],
        training_seed=int(job["training_seed"]),
        audit_request=_request_auditor(job["config"]),
    )
    if len(rows) != 1 or rows[0]["method"] != job["method"]:
        raise RuntimeError(f"Mission job {job['id']} returned unexpected rows")
    return {"rows": rows}


def _oracle_job(job):
    touch_status(job, "privileged oracle mission")
    from .oracle import run_oracle_episode

    env = _environment(
        job["config_for_env"],
        Scenario.from_dict(job["scenario"]),
        job["rollout_seed"],
        job["controller"],
        job["source_calibration"],
    )
    try:
        outcome = run_oracle_episode(
            env,
            job["delays"],
            replications=int(job["replications"]),
            max_decisions=job["max_decisions"],
            decision_interval_s=job["decision_interval_s"],
        )
    finally:
        _close(env)
    return {"outcome": outcome}


def dispatch_simulation_job(job):
    kind = job["kind"]
    if kind == "generate_episode":
        return _generate_episode_job(job)
    if kind == "calibration_episode":
        return _calibration_episode_job(job)
    if kind == "profile":
        return _profile_job(job)
    if kind == "mission":
        return _mission_job(job)
    if kind == "oracle":
        return _oracle_job(job)
    raise ValueError(f"Unknown simulation job {kind}")


def train_pipeline(config: dict, dataset, output) -> dict:
    output = _stage(output, config, "train_pipeline")
    manifest = {
        "controllers": {},
        "training_seeds": config["training_seeds"],
        "training_backend": config["backend"],
        "controller_comparison": "separate controller-conditioned data/model fits; independent residual and closed-loop threshold calibration for every controller",
        "manifest": str(output / "manifest.json"),
    }
    training_ablations = {"no_latency", "no_W", "no_reliability", "shuffled_pairs", "unpaired"}
    variants = ["timelyhelp"] + [a for a in config.get("ablations", []) if a in training_ablations]
    dataset_identity = _dataset_identity(dataset)
    for controller, (data, source_calibration) in _dataset_inputs(dataset).items():
        manifest["controllers"][controller] = {
            "source_calibration": source_calibration,
            "seeds": {},
        }
        for seed in config["training_seeds"]:
            fitted = {}
            for variant in variants:
                path = output / controller / f"seed_{seed}" / f"{variant}.pt"
                fit_identity = _fit_identity(
                    config, dataset_identity, variant, controller, seed, source_calibration
                )
                reuse_fit = _validated_fit_checkpoint(path, fit_identity)
                if reuse_fit and _variant_stamp_matches(path, config, dataset_identity, variant):
                    fitted[variant] = _checkpoint_record(path)
                    print(f"Reusing fitted {controller} seed {seed} {variant}", flush=True)
                    continue
                model_config = ModelConfig(
                    **config.get("learning", {})
                    | {
                        "seed": int(seed),
                        "feature_dim": int(data.histories.shape[-1]),
                        "time_scale": float(config.get("simulation", {}).get("horizon", 18.0)),
                        "omit_latency": variant == "no_latency",
                        "without_W": variant == "no_W",
                        "omit_reliability_metadata": variant == "no_reliability",
                    }
                )
                source_data = (
                    data.shuffled_pairs(seed)
                    if variant == "shuffled_pairs"
                    else _unpaired(data, seed)
                    if variant == "unpaired"
                    else data
                )
                fit_path = path.with_suffix(".fit.json")
                if reuse_fit:
                    print(
                        f"Reloading {controller} seed {seed} {variant} for calibration", flush=True
                    )
                    model = ConsequenceEnsemble.load(path)
                    fit_report = _read(fit_path)
                else:
                    print(f"Fitting {controller} seed {seed} {variant}", flush=True)
                    model = ConsequenceEnsemble(model_config)
                    fit = model.fit(source_data.split("train"), source_data.split("development"))
                    residual = model.calibrate(data.split("calibration"))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(path)
                    fit_report = {"fit": fit, "residual_calibration": residual}
                    write_json(fit_path, fit_report)
                    _write_fit_stamp(path, fit_identity)
                threshold = _threshold_calibration(
                    config,
                    model,
                    controller,
                    source_calibration,
                    model_path=path,
                    stage_output=output,
                    training_seed=seed,
                    variant=variant,
                )
                LearnedScheduler(
                    model, SchedulerConfig(**threshold["selected_configuration"])
                ).configure_thresholds(threshold)
                threshold_path = path.with_suffix(".selection.json")
                write_json(threshold_path, threshold)
                _write_profile(
                    config,
                    output,
                    model_path=path,
                    controller=controller,
                    source_calibration=source_calibration,
                    training_seed=seed,
                    variant=variant,
                )
                fitted[variant] = _checkpoint_record(path)
                _write_variant_stamp(path, config, dataset_identity, variant)
                write_json(
                    output / "progress.json",
                    {"completed_controller": controller, "training_seed": seed, "variant": variant},
                )
            main_path = Path(fitted["timelyhelp"]["model"])
            main_model = ConsequenceEnsemble.load(main_path)
            for variant in ("no_uncertainty", "no_hold"):
                if variant not in config.get("ablations", []):
                    continue
                selection_path = output / controller / f"seed_{seed}" / f"{variant}.selection.json"
                stamp_path = selection_path.with_suffix(".stamp.json")
                if _selection_stamp_matches(stamp_path, config, dataset_identity, variant):
                    fitted[variant] = {
                        **fitted["timelyhelp"],
                        "selection": str(selection_path),
                        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                        "weights_shared_with": "timelyhelp",
                    }
                    print(f"Reusing {controller} seed {seed} {variant} thresholds", flush=True)
                    continue
                threshold = _threshold_calibration(
                    config,
                    main_model,
                    controller,
                    source_calibration,
                    no_uncertainty=variant == "no_uncertainty",
                    no_hold=variant == "no_hold",
                    model_path=main_path,
                    stage_output=output,
                    training_seed=seed,
                    variant=variant,
                )
                write_json(selection_path, threshold)
                fitted[variant] = {
                    **fitted["timelyhelp"],
                    "selection": str(selection_path),
                    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    "weights_shared_with": "timelyhelp",
                }
                _write_selection_stamp(stamp_path, config, dataset_identity, variant)
            manifest["controllers"][controller]["seeds"][str(seed)] = fitted
            manifest["config_sha256"] = digest(config)
            manifest["dataset_sha256"] = dataset_identity
            manifest["code_sha256"] = code_hash("mission")
            write_json(output / "manifest.json", manifest)
    return {
        "models": str(output / "manifest.json"),
        "model_paths": manifest["controllers"],
        **manifest,
    }


def _models_manifest(models):
    if isinstance(models, dict):
        return models
    path = Path(models)
    return _read(path / "manifest.json" if path.is_dir() else path)


def _verified_gate(paths):
    """Load the recorded frozen gate only if its artifacts still match."""
    actual_weight_hash = hashlib.sha256(Path(paths["model"]).read_bytes()).hexdigest()
    if actual_weight_hash != paths["weight_sha256"]:
        raise ValueError("Frozen gate checkpoint differs from its training manifest")
    actual_selection_hash = hashlib.sha256(Path(paths["selection"]).read_bytes()).hexdigest()
    if paths.get("selection_sha256", actual_selection_hash) != actual_selection_hash:
        raise ValueError("Frozen gate selection thresholds differ from their training manifest")
    return ConsequenceEnsemble.load(paths["model"]), _read(paths["selection"])


def _request_auditor(config):
    limit = config.get("evaluation", {}).get(
        "request_audit_limit", 1 if config["mode"] == "smoke" else None
    )

    def audit(request, env, snapshot=None, policy=None):
        count = getattr(env, "_posthoc_request_audits", 0)
        if limit is not None and count >= int(limit):
            return None
        env._posthoc_request_audits = count + 1
        message = next(m for m in env.pending if m.request_id == request["request_id"])
        start = float(snapshot["time"])
        delay = (
            message.arrival_time - start
            if np.isfinite(message.arrival_time)
            else snapshot["config"].request_expiry
        )
        candidate = Candidate(request["source"], request["bridge"], max(0.0, delay))
        runner = CounterfactualRunner(env)
        row = runner.run(snapshot, candidate, replication=int(snapshot["replication"]))
        grid = sorted(set(config["generation"].get("delays", [0.0, 0.25, 0.75, 1.5, 3.0])))
        curve = [
            runner.run(
                snapshot,
                Candidate(request["source"], request["bridge"], tau),
                replication=int(snapshot["replication"]),
            )
            for tau in grid
        ]
        window = sampled_windows(
            grid,
            [r["info"] for r in curve],
            [r["outcomes"]["Q"]["collision"] for r in curve],
            risk_limit=config.get("calibration", {}).get("max_collision_rate", 0.1),
        )
        predicted_deadline = request.get("predicted_deadline")
        prediction_window = None
        if hasattr(policy, "model"):
            candidates = [Candidate(request["source"], request["bridge"], tau) for tau in grid]
            prediction = policy.model.predict(
                np.asarray(request["history"], dtype=np.float32), candidates
            )
            factor = 0.0 if policy.config.no_uncertainty else policy.config.uncertainty_weight
            prediction_window = sampled_windows(
                grid,
                prediction["info"] - factor * prediction["info_uncertainty"],
                np.minimum(
                    1.0,
                    prediction["collision"][:, 2]
                    + factor * prediction["collision_uncertainty"][:, 2],
                ),
                risk_limit=policy.config.risk_limit,
            )
            predicted_deadline = prediction_window["connected_deadline"]
        return {
            "info_effect": row["info"],
            "motion_effect": row["motion"],
            "actual_delay": delay,
            "source": request["source"],
            "request_id": request["request_id"],
            "request_status": message.reason,
            "predicted_deadline": predicted_deadline,
            "predicted_window": prediction_window,
            "actual_deadline": window["connected_deadline"],
            "right_censored": window["right_censored"],
            "window": window,
            "label_semantics": "isolated paired continuation with common nominal controller/fusion; future asks excluded",
            "audit_privilege": "posthoc snapshot and realized packet timing; never policy observation",
        }

    return audit


def _evaluation_identity(config, model_manifest, transfer):
    stable = {
        "controllers": model_manifest["controllers"],
        "training_seeds": model_manifest["training_seeds"],
        "training_backend": model_manifest.get("training_backend"),
    }
    return digest(
        {
            "config": digest(config),
            "models": digest(stable),
            "transfer": bool(transfer),
            "code": code_hash("mission"),
        }
    )


def _post_configure_query_weight(selection):
    config = SchedulerConfig(**selection["selected_configuration"])
    updated = asdict(config) | dict(selection["thresholds"])
    return float(SchedulerConfig(**updated).query_cost_weight)


def _method_specs(config, seed_paths):
    """Preserve the original comparator order: baselines, learned gates, cost sweep."""
    specs = []
    predictor = seed_paths["timelyhelp"]
    provenance = {
        "model": predictor["model"],
        "weight_sha256": predictor["weight_sha256"],
        "calibration_split": "calibration",
        "prediction_heads": "nominal A collision/completion only",
        "no_additional_training": True,
    }
    for name in config["evaluation"].get("methods", []):
        if name == "timelyhelp":
            continue
        if name == "failure_lead_time":
            specs.append(
                {
                    "role": "failure",
                    "method": name,
                    "model": predictor["model"],
                    "selection": predictor["selection"],
                    "weight_sha256": predictor["weight_sha256"],
                    "selection_sha256": predictor["selection_sha256"],
                    "predictor_provenance": provenance,
                }
            )
        else:
            specs.append({"role": "baseline", "method": name})
    for variant, paths in seed_paths.items():
        specs.append(
            {
                "role": "learned",
                "method": variant,
                "model": paths["model"],
                "selection": paths["selection"],
                "weight_sha256": paths["weight_sha256"],
                "selection_sha256": paths["selection_sha256"],
            }
        )
    selected_weight = _post_configure_query_weight(_read(predictor["selection"]))
    for penalty in config.get("evaluation", {}).get(
        "pareto_query_penalties",
        config.get("calibration", {}).get("query_penalties", [0.03]),
    ):
        if abs(float(penalty) - selected_weight) > 1e-12:
            specs.append(
                {
                    "role": "learned",
                    "method": f"timelyhelp_cost_{float(penalty):g}",
                    "model": predictor["model"],
                    "selection": predictor["selection"],
                    "weight_sha256": predictor["weight_sha256"],
                    "selection_sha256": predictor["selection_sha256"],
                    "query_cost_override": float(penalty),
                }
            )
    return specs


def _annotate_mission(row, job, model_manifest, transfer):
    original_method = row["method"]
    controller = job["controller"]
    seed = str(job["training_seed"])
    row["controller"] = controller
    row["method"] += f"/{controller}" if controller != "local" else ""
    row["frozen_gate"] = bool(transfer)
    row["training_backend"] = model_manifest.get("training_backend", job["config"]["backend"])
    model_variant = (
        "timelyhelp"
        if original_method.startswith("timelyhelp_cost_") or original_method == "failure_lead_time"
        else original_method
    )
    paths = model_manifest["controllers"][controller]["seeds"][seed].get(model_variant)
    row["gate_weight_sha256"] = paths["weight_sha256"] if paths else None
    row["gate_selection_sha256"] = paths.get("selection_sha256") if paths else None
    if original_method == "failure_lead_time":
        row["failure_predictor_provenance"] = job["policy"]["predictor_provenance"]
    return row


def _evaluation_jobs(
    config, config_for_env, output, model_manifest, scenarios, diagnostics, oracle_episode_budget
):
    mission_jobs = []
    oracle_jobs = {}
    oracle_appends = []
    for controller, controller_models in model_manifest["controllers"].items():
        source_calibration = controller_models["source_calibration"]
        for seed in model_manifest["training_seeds"]:
            sim_config = SimConfig.from_dict(
                config.get("simulation", {}) | {"controller": controller}
            )
            baseline = asdict(
                BaselineConfig(
                    sim=sim_config,
                    monte_carlo_samples=2 if config["mode"] == "smoke" else 16,
                    latency_support=1 if config["mode"] == "smoke" else 5,
                    risk_limit=config.get("calibration", {}).get("max_collision_rate", 0.1),
                    query_cost=sim_config.query_cost,
                )
            )
            specs = _method_specs(config, controller_models["seeds"][str(seed)])
            for index, scenario in enumerate(scenarios):
                rollout_seed = int(config.get("scenario_seed", 0)) + 30000 + index
                for spec in specs:
                    job_id = f"mission/{controller}/{seed}/{index}/{spec['method']}"
                    job = {
                        "kind": "mission",
                        "id": job_id,
                        "config": config,
                        "config_for_env": config_for_env,
                        "controller": controller,
                        "training_seed": int(seed),
                        "method": spec["method"],
                        "policy": spec,
                        "baseline": baseline,
                        "scenario": scenario.to_dict(),
                        "scenario_index": index,
                        "rollout_seed": rollout_seed,
                        "source_calibration": source_calibration,
                        "status_dir": str(status_directory(output)),
                        "shard": str(
                            Path(output)
                            / "shards"
                            / "missions"
                            / controller
                            / f"seed_{seed}"
                            / f"{index:04d}_{spec['method']}.json"
                        ),
                    }
                    job["work_sha256"] = work_hash(
                        "mission",
                        config_for_env,
                        {
                            "id": job_id,
                            "method": spec["method"],
                            "policy": spec,
                            "rollout_seed": rollout_seed,
                            "scenario": job["scenario"],
                            "baseline": baseline,
                            "source_calibration": source_calibration,
                        },
                    )
                    mission_jobs.append(job)
                if diagnostics.get("oracle_enabled", False) and index < oracle_episode_budget:
                    oracle_id = f"oracle/{controller}/{index}"
                    if oracle_id not in oracle_jobs:
                        oracle_job = {
                            "kind": "oracle",
                            "id": oracle_id,
                            "config": config,
                            "config_for_env": config_for_env,
                            "controller": controller,
                            "scenario": scenario.to_dict(),
                            "scenario_index": index,
                            "rollout_seed": rollout_seed,
                            "source_calibration": source_calibration,
                            "delays": list(config["generation"]["delays"]),
                            "replications": int(diagnostics.get("oracle_replications", 1)),
                            "max_decisions": diagnostics.get("oracle_max_decisions", 1),
                            "decision_interval_s": diagnostics.get("oracle_decision_interval_s"),
                            "status_dir": str(status_directory(output)),
                            "shard": str(
                                Path(output)
                                / "shards"
                                / "oracle"
                                / controller
                                / f"{index:04d}.json"
                            ),
                        }
                        oracle_job["work_sha256"] = work_hash(
                            "oracle",
                            config_for_env,
                            {
                                "id": oracle_id,
                                "rollout_seed": rollout_seed,
                                "scenario": oracle_job["scenario"],
                                "delays": oracle_job["delays"],
                                "replications": oracle_job["replications"],
                                "max_decisions": oracle_job["max_decisions"],
                                "decision_interval_s": oracle_job["decision_interval_s"],
                                "source_calibration": source_calibration,
                            },
                        )
                        oracle_jobs[oracle_id] = oracle_job
                    oracle_appends.append(
                        {
                            "job_id": oracle_id,
                            "controller": controller,
                            "training_seed": int(seed),
                            "rollout_seed": rollout_seed,
                            "scenario": scenario.to_dict(),
                        }
                    )
    return mission_jobs, oracle_jobs, oracle_appends


def evaluate_pipeline(config: dict, models, output, dataset=None, *, transfer=False) -> dict:
    output = _stage(output, config, "transfer" if transfer else "evaluate_pipeline")
    model_manifest = _models_manifest(models)
    count = int(config["evaluation"]["transfer_episodes" if transfer else "episodes"])
    scenarios = (
        _transfer_scenarios(config, count) if transfer else _scenario_list(config, "test", count)
    )
    diagnostics = config.get("diagnostics", {})
    oracle_episode_budget = min(count, int(diagnostics.get("oracle_episodes", count)))
    config_for_env = deepcopy(config)
    if transfer:
        config_for_env["backend"] = "gazebo"
    backend_protocol = None
    if config_for_env["backend"] != "reference":
        from .backends.native import native_backend_protocol

        backend_protocol = native_backend_protocol(config_for_env["backend"])
    resume_identity = _evaluation_identity(config, model_manifest, transfer)
    if (output / "manifest.json").is_file():
        existing = _read(output / "manifest.json")
        if existing.get("resume_identity") == resume_identity:
            print("Evaluation manifest is complete; not repeating missions.", flush=True)
            return existing
        raise ValueError(
            f"{output / 'manifest.json'} belongs to a different evaluation revision; "
            "use a fresh output directory"
        )
    mission_jobs, oracle_jobs, oracle_appends = _evaluation_jobs(
        config,
        config_for_env,
        output,
        model_manifest,
        scenarios,
        diagnostics,
        oracle_episode_budget,
    )
    scheduled = mission_jobs + list(oracle_jobs.values())
    print(
        f"Evaluating {len(mission_jobs)} missions and {len(oracle_jobs)} privileged oracle episodes",
        flush=True,
    )
    finished = {
        job["id"]: result for job, result in zip(scheduled, execute_jobs(config_for_env, scheduled))
    }
    records = []
    for job in mission_jobs:
        row = deepcopy(finished[job["id"]]["rows"][0])
        records.append(_annotate_mission(row, job, model_manifest, transfer))
    write_json(output / "records.json", records)
    oracle_records = []
    for item in oracle_appends:
        result = deepcopy(finished[item["job_id"]]["outcome"])
        scenario = item["scenario"]
        controller = item["controller"]
        oracle_records.append(
            {
                "record_type": "privileged_mission",
                "privileged": True,
                "method": "privileged_oracle" + (f"/{controller}" if controller != "local" else ""),
                "training_seed": item["training_seed"],
                "rollout_seed": item["rollout_seed"],
                "episode_id": f"{scenario['layout_id']}:0:{item['rollout_seed']}",
                "controller": controller,
                "backend": config_for_env["backend"],
                "backend_protocol": backend_protocol,
                "oracle_shared_across_training_seeds": True,
                **{
                    name: scenario[name]
                    for name in ("layout_id", "split", "setting", "family", "disturbances")
                },
                **result,
            }
        )
    if oracle_records:
        write_json(output / "privileged_oracle_records.json", oracle_records)
    oracle_cache = oracle_jobs
    frozen = {
        controller: {
            seed: paths["timelyhelp"]["weight_sha256"] for seed, paths in c["seeds"].items()
        }
        for controller, c in model_manifest["controllers"].items()
    }
    manifest = {
        "records": str(output / "records.json"),
        "missions": len(records),
        "privileged_oracle_records": str(output / "privileged_oracle_records.json")
        if oracle_records
        else None,
        "privileged_oracle_mission_records": len(oracle_records),
        "distinct_oracle_missions_executed": len(oracle_cache),
        "oracle_protocol": diagnostics,
        "oracle_scenario_indices": list(range(oracle_episode_budget)) if oracle_records else [],
        "oracle_comparison_scope": "the declared bounded scenario subset; matched ordinary mission keys only",
        "training_seeds": model_manifest["training_seeds"],
        "scenario_manifest": [s.to_dict() for s in scenarios],
        "transfer": transfer,
        "backend": config_for_env["backend"],
        "frozen_gate_weights": frozen,
        "backend_protocol": backend_protocol,
        "frozen_gate_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest(),
        "transfer_adaptation": "No model retraining, source-prior recalibration or selection-threshold fitting"
        if transfer
        else None,
        "distinct_base_layouts": sorted({scenario.layout_id for scenario in scenarios}),
        "layout_geometry_semantics": "three declared base corridor/junction layouts for transfer; physical occlusion/source-view disturbances may modify realized geometry",
        "realized_scenario_geometry_hashes": sorted(
            {
                hashlib.sha256(json.dumps(s.geometry(), sort_keys=True).encode()).hexdigest()
                for s in scenarios
            }
        ),
        "transfer_layout_coverage": (
            "all three declared layouts"
            if len({s.layout_id for s in scenarios}) == 3
            else "bounded subset of three declared layouts"
        )
        if transfer
        else None,
        "request_audit_limit_per_mission": config.get("evaluation", {}).get(
            "request_audit_limit", 1 if config["mode"] == "smoke" else None
        ),
        "resume_identity": resume_identity,
        "config_sha256": digest(config),
        "code_sha256": code_hash("mission"),
        "parallel_gpus": list(configured_gpus()) if parallel_simulations(config_for_env) else [],
        "scheduling": "matched missions stay intact; methods may occupy different GPUs but share rollout seeds",
        "manifest": str(output / "manifest.json"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _best_mean_no_query_motion(labels):
    """Best expected motion effect; sources sharing W do not add samples."""
    controls = {}
    for row in labels:
        controls.setdefault((row["bridge"], row["delay"]), {}).setdefault(
            row.get("replication", 0),
            row["outcomes"]["A"]["cost"] - row["outcomes"]["W"]["cost"],
        )
    return max(
        [0.0] + [float(np.mean(list(replications.values()))) for replications in controls.values()]
    )


def _privileged_mission_report(records, oracle_records, *, samples, seed):
    """Compare actual oracle missions separately from deployable-method results."""
    if not oracle_records:
        return {"status": "privileged oracle missions were not executed", "cohorts": {}}
    cohorts = {}
    for backend, controller in sorted({(r["backend"], r["controller"]) for r in oracle_records}):
        selected_oracle = [
            r for r in oracle_records if r["backend"] == backend and r["controller"] == controller
        ]
        matched_keys = {
            (r["training_seed"], r["layout_id"], r["episode_id"]) for r in selected_oracle
        }
        controls = [
            r
            for r in records
            if r["backend"] == backend
            and r["controller"] == controller
            and (r["training_seed"], r["layout_id"], r["episode_id"]) in matched_keys
            and r["method"].split("/")[0]
            in {"nominal", "cautious", "no_query_nominal", "no_query_cautious", "model_based_voi"}
        ]
        # They are complete executed missions. Re-type solely for the existing
        # mission bootstrap in this explicitly privileged diagnostic, retaining
        # the privilege flag and leaving original records untouched.
        diagnostic_rows = controls + [dict(r, record_type="mission") for r in selected_oracle]
        summaries = method_summary(diagnostic_rows)
        oracle_method = selected_oracle[0]["method"]
        comparisons = {
            method: {
                metric: hierarchical_bootstrap(
                    diagnostic_rows,
                    metric,
                    comparison=(oracle_method, method),
                    samples=samples,
                    seed=seed,
                )
                for metric in ("success", "collision", "elapsed_time", "queries")
            }
            for method in sorted({r["method"] for r in controls})
        }
        frontier = pareto_frontier(summaries)
        cohorts[f"{backend}/{controller}"] = {
            "summary": summaries,
            "diagnostic_frontier": frontier,
            "oracle_on_diagnostic_frontier": any(
                row["method"] == oracle_method for row in frontier
            ),
            "paired_comparisons": comparisons,
            "actual_oracle_rollouts": len(
                {(r["layout_id"], r["episode_id"]) for r in selected_oracle}
            ),
            "comparison_scope": "ordinary methods restricted to exactly the oracle's matched scenario/rollout subset",
        }
    return {
        "status": "separate privileged executed-mission headroom diagnostic",
        "privilege": "full-state counterfactual outcomes and selected forced arrival time; source payloads remain imperfect/bounded",
        "inference_units": "complete matched missions; counterfactual sibling labels excluded; oracle runs reused across fitted-model seeds",
        "limitations": "Finite Monte Carlo and replanning oracle is an idealized diagnostic, not a guaranteed performance upper bound or deployable method",
        "cohorts": cohorts,
    }


def report_pipeline(
    config: dict, records, output, counterfactual_rows=None, models=None, *, privileged_records=None
):
    output = _stage(output, config, "report_pipeline")
    oracle_records = []
    if isinstance(records, dict):
        oracle_path = records.get("privileged_oracle_records")
        if oracle_path:
            oracle_records = _read(oracle_path)
        records = _read(records["records"])
    elif isinstance(records, (str, Path)):
        oracle_path = Path(records).with_name("privileged_oracle_records.json")
        if oracle_path.is_file():
            oracle_records = _read(oracle_path)
        records = _read(records)
    if privileged_records is not None:
        oracle_records = (
            _read(privileged_records)
            if isinstance(privileged_records, (str, Path))
            else list(privileged_records)
        )
    if isinstance(counterfactual_rows, dict):
        rows = _read(counterfactual_rows["records"])
    elif isinstance(counterfactual_rows, (str, Path)):
        path = Path(counterfactual_rows)
        if path.is_dir():
            rows = _read(_read(path / "manifest.json")["records"])
        elif path.suffix == ".npz":
            name = path.stem.removeprefix("paired_")
            rows = _read(path.with_name(f"records_{name}.json"))
        else:
            raw = _read(path)
            rows = _read(raw["records"]) if isinstance(raw, dict) else raw
    else:
        rows = counterfactual_rows or []
    local_rows = [row for row in rows if row.get("controller", "local") == "local"]
    predictions = []
    if models is not None:
        manifest = _models_manifest(models)
        local = manifest["controllers"].get("local", next(iter(manifest["controllers"].values())))
        test_rows = [row for row in local_rows if row["split"] == "test"]
        grid_dataset = PairedDataset.from_records(test_rows) if test_rows else None
        window_grid = []
        for training_seed in manifest["training_seeds"]:
            model, threshold = _verified_gate(local["seeds"][str(training_seed)]["timelyhelp"])
            operating_point = threshold["selected_configuration"]
            risk_limit = operating_point["risk_limit"]
            uncertainty_weight = operating_point["uncertainty_weight"]
            if grid_dataset is not None:
                window_grid.extend(
                    dict(
                        entry,
                        training_seed=int(training_seed),
                        risk_limit=risk_limit,
                        uncertainty_weight=uncertainty_weight,
                        operating_point="selected independent closed-loop calibration",
                    )
                    for entry in window_prediction_grid(
                        model,
                        grid_dataset,
                        config["generation"]["delays"],
                        risk_limit=risk_limit,
                        uncertainty_weight=uncertainty_weight,
                    )
                )
            seen = set()
            for row in test_rows:
                key = (
                    row["layout"],
                    row["rollout_id"],
                    row["snapshot_id"],
                    row["source"],
                    row["bridge"],
                    row["delay"],
                )
                if key in seen:
                    continue
                seen.add(key)
                prediction = model.predict(
                    np.asarray(row["history"], np.float32),
                    [Candidate(row["source"], row["bridge"], row["delay"])],
                )
                predictions.append(
                    {
                        k: row[k]
                        for k in (
                            "layout",
                            "rollout_id",
                            "snapshot_id",
                            "source",
                            "bridge",
                            "delay",
                        )
                    }
                    | {
                        "training_seed": int(training_seed),
                        "prediction_split": "test",
                        "risk_limit": risk_limit,
                        "uncertainty_weight": uncertainty_weight,
                        "operating_point": "selected independent closed-loop calibration",
                        "predicted_info": float(
                            prediction["info"][0]
                            - uncertainty_weight * prediction["info_uncertainty"][0]
                        ),
                        "predicted_collision": float(
                            min(
                                1.0,
                                prediction["collision"][0, 2]
                                + uncertainty_weight * prediction["collision_uncertainty"][0, 2],
                            )
                        ),
                    }
                )
        write_json(output / "useful_window_grid.json", window_grid)
    write_json(output / "window_predictions.json", predictions)
    scene_audit = (
        matched_observable_scenes(local_rows)
        if local_rows
        else {"identical_confidence_pairs": [], "identical_observation_aliases": []}
    )
    write_json(output / "matched_observable_scenes.json", scene_audit)
    report_config = deepcopy(config)
    gate_hashes = {
        f"{r['controller']}:{r['training_seed']}:{r['method']}": r["gate_weight_sha256"]
        for r in records
        if r.get("frozen_gate") and r.get("gate_weight_sha256")
    }
    if gate_hashes:
        report_config["frozen_gate_sha256"] = hashlib.sha256(
            json.dumps(gate_hashes, sort_keys=True).encode()
        ).hexdigest()
    artifacts = generate_report(
        records,
        local_rows,
        output,
        config=report_config,
        bootstrap_samples=int(config["evaluation"].get("bootstrap_samples", 1000)),
        seed=int(config.get("scenario_seed", 0)),
        window_predictions=predictions,
        scene_audit=scene_audit,
    )
    # Hindsight headroom uses isolated counterfactual labels, separately from all
    # online-policy mission results and never included in mission bootstrap.
    groups = {}
    for row in local_rows:
        groups.setdefault((row["layout"], row["rollout_id"], row["snapshot_id"]), []).append(row)
    headroom = []
    oracle = DiagnosticOracle()
    for key, labels in groups.items():
        decision = oracle.select_counterfactuals(
            labels, query_cost=config.get("simulation", {}).get("query_cost", 0.03), risk_limit=1.0
        )
        best_caution = _best_mean_no_query_motion(labels)
        headroom.append(
            {
                "layout": key[0],
                "rollout_id": key[1],
                "snapshot_id": key[2],
                "oracle_source": decision.source,
                "oracle_bridge": decision.bridge,
                "oracle_delay": decision.delay,
                "oracle_net": decision.net_value,
                "best_no_query_motion_net": best_caution,
                "headroom_over_caution": decision.net_value - best_caution,
                "cost_estimator": "mean Monte Carlo outcomes per candidate; reused W controls counted once per replication",
                "comparison_semantics": "weighted-cost headroom over best mean no-query motion; safety frontier assessed separately from mission records",
                "privileged": True,
                "status": "hindsight isolated-snapshot diagnostic; not an online mission policy",
            }
        )
    write_json(output / "privileged_oracle_headroom.json", headroom)
    write_json(
        output / "privileged_oracle_mission_report.json",
        _privileged_mission_report(
            records,
            oracle_records,
            samples=int(config["evaluation"].get("bootstrap_samples", 1000)),
            seed=int(config.get("scenario_seed", 0)),
        ),
    )
    return {
        **artifacts,
        "oracle_headroom": str(output / "privileged_oracle_headroom.json"),
        "oracle_mission_diagnostics": str(output / "privileged_oracle_mission_report.json"),
    }


@contextmanager
def _pipeline_output_lock(output):
    """Keep one writer for a pipeline directory, including across SSH sessions.

    The kernel lock, rather than the informational owner PID, defines ownership.
    Keep the file after release: unlinking it could let two processes lock
    different inodes for the same output directory.
    """
    import fcntl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.seek(0)
            owner = lock.read().strip() or "owner information unavailable"
            raise RuntimeError(
                f"Pipeline output directory {output} is already locked by another run: {owner}. "
                "Attach to the existing run or choose a separate output directory."
            ) from error
        try:
            owner = {
                "pid": os.getpid(),
                "started_unix_time": time.time(),
                "output": str(output.resolve()),
                "ownership": "advisory flock; PID is informational and may remain after release",
            }
            lock.seek(0)
            lock.truncate()
            lock.write(json.dumps(owner, sort_keys=True) + "\n")
            lock.flush()
            yield output
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_pipeline(config: dict, output) -> dict:
    with _pipeline_output_lock(output) as locked_output:
        return _unlocked_run_pipeline(config, locked_output)


def _unlocked_run_pipeline(config: dict, output) -> dict:
    import traceback

    output = _stage(output, config, "run_pipeline")
    status = {
        "pid": os.getpid(),
        "backend": config["backend"],
        "mode": config["mode"],
        "parallel_gpus": list(configured_gpus()) if parallel_simulations(config) else [],
        "scheduling": "single pilot partitioned across isolated Isaac workers",
        "started_unix_time": time.time(),
        "history": [],
    }

    def record_stage(stage, *, error=None, resumable_stage=None):
        now = time.time()
        state = stage if stage in {"complete", "failed", "interrupted"} else "running"
        status.update(
            stage=stage,
            status=state,
            unix_time=now,
            updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            resumable_stage=resumable_stage,
        )
        status["history"].append({"stage": stage, "unix_time": now})
        if error is not None:
            status["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        if state in {"complete", "failed", "interrupted"}:
            status["finished_unix_time"] = now
        write_json(output / "run_status.json", status)

    record_stage("starting", resumable_stage="generate")
    print(
        f"TimelyHelp {config['mode']} pipeline pid={os.getpid()} backend={config['backend']}",
        flush=True,
    )
    active_stage = "generate"
    try:
        record_stage(active_stage, resumable_stage=active_stage)
        data = generate_data(config, output / "data")
        active_stage = "train"
        record_stage(active_stage, resumable_stage=active_stage)
        models = train_pipeline(config, data, output / "models")
        active_stage = "evaluate"
        record_stage(active_stage, resumable_stage=active_stage)
        evaluation = evaluate_pipeline(config, models, output / "evaluation", dataset=data)
        active_stage = "report"
        record_stage(active_stage, resumable_stage=active_stage)
        report = report_pipeline(
            config, evaluation, output / "report", counterfactual_rows=data, models=models
        )
        manifest = {
            "data": data,
            "models": models,
            "evaluation": evaluation,
            "report": report,
            "mode": config["mode"],
            "backend": config["backend"],
            "full_scale_experiments_executed": config["mode"] == "confirmatory",
        }
        write_json(output / "pipeline_manifest.json", manifest)
        record_stage("complete")
    except BaseException as error:
        state = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record_stage(state, error=error, resumable_stage=active_stage)
        raise
    return manifest
