"""Matched closed-loop evaluation and cluster-aware descriptive inference.

Only mission rollouts enter inferential summaries. Counterfactual sibling branches
are labels, not additional experimental subjects. A smoke suite is never treated
as confirmatory evidence, even when its bootstrap interval happens to be narrow.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from inspect import signature
from typing import Any, Callable, Iterable, Mapping
import time

import numpy as np

from .types import Candidate


def _public_metadata(observation: dict) -> dict:
    metadata = deepcopy(observation.get("sources", {}))
    for source in metadata.values():
        source.setdefault("latency_samples", source.get("latencies", []))
        source.setdefault(
            "available", source.get("status", "online") in ("online", "available", "pending")
        )
    metadata["remaining_query_budget"] = int(observation.get("remaining_queries", 0))
    return metadata


def _select(policy: Any, history: np.ndarray, observation: dict):
    method = policy.select
    if "observation" in signature(method).parameters:
        return method(history, _public_metadata(observation), observation=observation)
    return method(history, _public_metadata(observation))


def _new_policy(policy: Any, seed: int):
    if callable(policy) and not hasattr(policy, "select"):
        parameters = signature(policy).parameters
        result = policy(seed=seed) if "seed" in parameters else policy()
    else:
        result = deepcopy(policy)
    if hasattr(result, "reset"):
        result.reset(seed)
    return result


def _scenario_metadata(scenario: Any) -> dict:
    data = asdict(scenario) if is_dataclass(scenario) else dict(scenario)
    return {
        key: data.get(key) for key in ("layout_id", "split", "setting", "family", "disturbances")
    }


def run_episode(
    env: Any,
    policy: Any,
    *,
    max_steps: int | None = None,
    commitment_s: float | None = None,
    audit_request: Callable | None = None,
    step_callback: Callable | None = None,
) -> dict:
    """Run one finite-commitment scheduler through the common environment.

    New requests are suppressed while a noncancelled request is pending. At the
    next commitment tick the scheduler may keep, cancel, or replace that request.
    Only public pending metadata are consulted. Environment fusion performs the
    shared capture-age, expiration, supersession and arrival admission checks.
    ``audit_request`` is an optional *post hoc* privileged evaluator; its output
    is attached to the report and never enters a policy observation.
    ``step_callback(env, telemetry)`` supports diagnostic viewers after completed
    steps. Its return value is ignored and it is excluded from selection timing.
    """
    config = env.config
    dt = float(config.dt)
    commitment_s = float(commitment_s or config.commitment)
    if commitment_s <= 0:
        raise ValueError("Commitment must be positive")
    steps_limit = max_steps if max_steps is not None else int(np.ceil(config.horizon / dt)) + 2
    if steps_limit < 1:
        raise ValueError("max_steps must be positive")
    commitment_steps = max(1, int(np.ceil(commitment_s / dt)))
    current_bridge = "nominal"
    decisions, selection_seconds, requests = [], [], []
    audit_snapshots = []
    request_bridges = {}
    issuing_bridge = None
    if hasattr(policy, "reassess_packet"):

        def admission_filter(packet, observation):
            # Payload truth/corruption flags are never forwarded to the gate.
            headers = {
                key: packet[key] for key in ("source", "capture_time", "arrival_time", "request_id")
            }
            headers["bridge"] = request_bridges.get(
                packet["request_id"], issuing_bridge or "cautious"
            )
            current_history = (
                env.current_history_features()
                if hasattr(env, "current_history_features")
                else env.history_features()
            )
            kwargs = {}
            if "observation" in signature(policy.reassess_packet).parameters:
                kwargs["observation"] = observation
            if "latest_capture_time" in signature(policy.reassess_packet).parameters:
                kwargs["latest_capture_time"] = env.evidence.last_capture.get(packet["source"])
            result = policy.reassess_packet(
                current_history,
                headers,
                float(observation["time"]),
                _public_metadata(observation),
                **kwargs,
            )
            return bool(result.get("accept", False))

        env.admission_filter = admission_filter
    steps = 0
    while not env.done and steps < steps_limit:
        if steps % commitment_steps == 0:
            observation = env.observe()
            before = time.perf_counter()
            decision = _select(policy, env.history_features(), observation)
            selection_seconds.append(time.perf_counter() - before)
            bridge = getattr(decision, "bridge", "cautious")
            source = getattr(decision, "source", "none")
            if bridge not in ("nominal", "cautious", "hold"):
                raise ValueError(f"Invalid scheduler bridge {bridge!r}")
            pending = [
                request
                for request in observation.get("pending", [])
                if request.get("reason", "pending") == "pending"
                and not request.get("cancelled", False)
            ]
            # Some backends call the public queue requests; do not inspect env.pending.
            if not pending:
                pending = [
                    request
                    for request in observation.get("requests", [])
                    if request.get("reason", "pending") == "pending"
                    and not request.get("cancelled", False)
                ]
            for request in pending:
                if source == "none" or request["source"] != source:
                    env.cancel(request["request_id"])
            same_pending = any(request["source"] == source for request in pending)
            requested = False
            if (
                source != "none"
                and not same_pending
                and observation.get("remaining_queries", 0) > 0
            ):
                candidate = Candidate(
                    source, bridge, max(0.0, float(getattr(decision, "delay", 0.0)))
                )
                # Delay conditions selection; runtime draws latency from the source,
                # rather than supplying the model's mean as an oracle override.
                audit_snapshot = env.snapshot() if audit_request is not None else None
                if audit_snapshot is not None:
                    # Labels concern the fixed nominal controller/common adapter.
                    # Do not copy a runtime closure capturing this live scheduler.
                    audit_snapshot["admission_filter"] = None
                issuing_bridge = bridge
                result = env.request(candidate)
                if result is not None:
                    request_bridges[result] = bridge
                issuing_bridge = None
                requested = result is not None
                if requested:
                    requests.append(
                        {
                            "time": float(observation["time"]),
                            "source": source,
                            "bridge": bridge,
                            "predicted_delay": candidate.delay,
                            "request_id": result,
                            "predicted_deadline": getattr(decision, "deadline", None),
                            "history": env.history_features().tolist(),
                            "observation": deepcopy(observation),
                        }
                    )
                    audit_snapshots.append(audit_snapshot)
            current_bridge = bridge
            record = asdict(decision) if is_dataclass(decision) else dict(vars(decision))
            record.update(time=float(observation["time"]), requested=requested)
            decisions.append(record)
        env.advance(current_bridge)
        steps += 1
        if step_callback is not None:
            step_callback(env, {"step": steps, "bridge": current_bridge, "decision": decisions[-1]})
    outcomes = dict(env.outcomes())
    completed = bool(outcomes.get("completed", outcomes.get("completion", False)))
    collision = bool(outcomes.get("collision", False))
    elapsed = float(outcomes.get("time", outcomes.get("elapsed_time", steps * dt)))
    within_budget = elapsed <= float(config.horizon) + dt * 1e-6
    truncated = not env.done
    outcomes.update(
        completed=completed,
        collision=collision,
        success=bool(completed and not collision and within_budget and not truncated),
        within_budget=within_budget,
        truncated=truncated,
        steps=steps,
        completion_time=elapsed if completed else None,
        elapsed_time=elapsed,
        selection_seconds_mean=float(np.mean(selection_seconds)) if selection_seconds else 0.0,
        selection_seconds_max=float(max(selection_seconds, default=0.0)),
        decisions=decisions,
        request_decisions=requests,
    )
    # A rejected stale reply is measurable online; a useless answer requires labels.
    observation = env.observe()
    messages = observation.get("messages", [])
    public_pending = observation.get("pending", [])
    outcomes.setdefault(
        "late_requests",
        sum(
            m.get("reason") in ("stale", "expired", "newer_belief", "newer_source_belief")
            for m in messages
        )
        + sum(m.get("reason") == "expired" for m in public_pending),
    )
    delays = outcomes.get("reply_delays", [])
    outcomes.setdefault("reply_delay", float(np.mean(delays)) if delays else None)
    outcomes.setdefault("supervisor_workload", outcomes.get("workload"))
    outcomes.setdefault("useless_requests_audited", None)
    outcomes.setdefault("deadline_error", None)
    if audit_request is not None:
        audits = []
        for request, snapshot in zip(requests, audit_snapshots):
            kwargs = {}
            if "snapshot" in signature(audit_request).parameters:
                kwargs["snapshot"] = snapshot
            if "policy" in signature(audit_request).parameters:
                kwargs["policy"] = policy
            audits.append(audit_request(request, env, **kwargs))
        outcomes["request_audits"] = audits
        usable = [audit for audit in audits if audit is not None]
        outcomes["useless_requests_audited"] = sum(
            float(audit["info_effect"]) <= 0 for audit in usable
        )
        outcomes["ineffective_requests_audited"] = sum(
            float(audit["info_effect"]) <= 0
            or audit.get("request_status")
            in ("cancelled", "expired", "stale", "predicted_useless", "unknown")
            for audit in usable
        )
        outcomes["audited_requests"] = len(usable)
        outcomes["request_audit_coverage"] = len(usable) / len(requests) if requests else None
        errors = [
            abs(float(audit["predicted_deadline"]) - float(audit["actual_deadline"]))
            for audit in usable
            if audit.get("predicted_deadline") is not None
            and audit.get("actual_deadline") is not None
            and not audit.get("right_censored", False)
        ]
        outcomes["deadline_error"] = float(np.mean(errors)) if errors else None
    return outcomes


def run_evaluation(
    env_factory: Callable,
    policies: Mapping[str, Any],
    scenarios: Iterable,
    seeds: Iterable[int] = (0,),
    *,
    training_seed: int = 0,
    max_steps: int | None = None,
    commitment_s: float | None = None,
    audit_request: Callable | None = None,
) -> list[dict]:
    """Matched scenario/environment randomness across methods, fresh policy state.

    ``training_seed`` identifies an independent learned model fit; ``seeds`` are
    rollout randomness. Increasing the latter does not create training seeds.
    Factories receive ``seed`` when accepted. The backend owns exogenous RNGs;
    all methods receive exactly the same scenario and rollout seed.
    """
    scenarios, seeds = list(scenarios), list(seeds)
    records = []
    for scenario_index, scenario in enumerate(scenarios):
        metadata = _scenario_metadata(scenario)
        for rollout_seed in seeds:
            episode_id = f"{metadata['layout_id']}:{scenario_index}:{rollout_seed}"
            for name, prototype in policies.items():
                policy = _new_policy(prototype, training_seed)
                env = env_factory(scenario, int(rollout_seed))
                try:
                    result = run_episode(
                        env,
                        policy,
                        max_steps=max_steps,
                        commitment_s=commitment_s,
                        audit_request=audit_request,
                    )
                finally:
                    close = getattr(env, "close", None)
                    if close is not None:
                        close()
                records.append(
                    {
                        "record_type": "mission",
                        "method": name,
                        "training_seed": int(training_seed),
                        "rollout_seed": int(rollout_seed),
                        "episode_id": episode_id,
                        "backend": getattr(env, "backend_name", "reference"),
                        "backend_protocol": getattr(env, "backend_protocol", None),
                        **metadata,
                        **result,
                    }
                )
    return records


def _matched_clusters(records: list[dict], metric: str, methods: list[str]):
    cells: dict[tuple, dict[str, float]] = {}
    for record in records:
        if record.get("record_type", "mission") != "mission":
            raise ValueError("Counterfactual siblings cannot enter mission bootstrap")
        value = record.get(metric)
        if value is None or not np.isfinite(float(value)):
            continue
        key = (record["training_seed"], record["layout_id"], record["episode_id"])
        if record["method"] in cells.setdefault(key, {}):
            raise ValueError("Duplicate method for a matched mission")
        cells[key][record["method"]] = float(value)
    clusters: dict[Any, dict[Any, dict[Any, np.ndarray]]] = {}
    for (seed, layout, episode), values in cells.items():
        if all(method in values for method in methods):
            clusters.setdefault(seed, {}).setdefault(layout, {})[episode] = np.array(
                [values[m] for m in methods]
            )
    return clusters


def hierarchical_bootstrap(
    records: list[dict],
    metric: str,
    *,
    methods: list[str] | None = None,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    comparison: tuple[str, str] | None = None,
) -> dict:
    """Paired crossed seed/layout/complete-rollout bootstrap.

    Each draw uses identical cluster indices for every method. Within a sampled
    layout, complete matched mission tuples are sampled; sibling branch rows are
    rejected. Missing measurements are excluded as matched tuples, never filled.
    With one seed/layout this remains descriptive and cannot estimate those
    population variance components.
    """
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap parameters")
    backends = {record.get("backend", "reference") for record in records}
    if len(backends) > 1:
        raise ValueError(
            "Mission bootstrap requires one backend; stratify native simulator cohorts"
        )
    methods = methods or sorted({record["method"] for record in records})
    if comparison:
        methods = list(comparison)
    clusters = _matched_clusters(records, metric, methods)
    if not clusters:
        return {"metric": metric, "methods": methods, "status": "no matched finite observations"}
    seeds = list(clusters)
    layouts = sorted(set.intersection(*(set(clusters[s]) for s in seeds)), key=str)
    # Test scenarios/randomness are shared across training seeds, so layout and
    # rollout resampling must also be shared. Repeated baseline outcomes across
    # seeds are not additional independent environmental observations.
    episode_keys = {
        layout: sorted(set.intersection(*(set(clusters[s][layout]) for s in seeds)), key=str)
        for layout in layouts
    }
    layouts = [layout for layout in layouts if episode_keys[layout]]
    if not layouts:
        return {
            "metric": metric,
            "methods": methods,
            "status": "no layouts/complete rollouts matched across training seeds",
        }
    means = []
    for training_seed in seeds:
        means.append(
            np.mean(
                [
                    np.mean(
                        [
                            clusters[training_seed][layout][episode]
                            for episode in episode_keys[layout]
                        ],
                        axis=0,
                    )
                    for layout in layouts
                ],
                axis=0,
            )
        )
    estimate = np.mean(means, axis=0)
    rng = np.random.default_rng(seed)
    draws = np.zeros((samples, len(methods)))
    for draw in range(samples):
        selected_layouts = rng.integers(0, len(layouts), len(layouts))
        selected_episodes = {
            layout: rng.integers(0, len(episode_keys[layout]), len(episode_keys[layout]))
            for layout in layouts
        }
        seed_means = []
        for seed_index in rng.integers(0, len(seeds), len(seeds)):
            layout_means = []
            for layout_index in selected_layouts:
                layout = layouts[layout_index]
                chosen = selected_episodes[layout]
                layout_means.append(
                    np.mean(
                        [
                            clusters[seeds[seed_index]][layout][episode_keys[layout][index]]
                            for index in chosen
                        ],
                        axis=0,
                    )
                )
            seed_means.append(np.mean(layout_means, axis=0))
        draws[draw] = np.mean(seed_means, axis=0)
    alpha = (1.0 - confidence) / 2
    independent_layouts = len(layouts)
    result = {
        "metric": metric,
        "methods": methods,
        "estimate": dict(zip(methods, estimate.tolist())),
        "interval": {
            method: np.quantile(draws[:, index], [alpha, 1 - alpha]).tolist()
            for index, method in enumerate(methods)
        },
        "independent_training_seeds": len(seeds),
        "layouts": independent_layouts,
        "matched_missions": len(seeds) * sum(len(episode_keys[layout]) for layout in layouts),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "weighting": "equal training seeds/layouts; crossed layout and complete-rollout draws shared across seeds and methods",
        "status": "descriptive; insufficient independent clusters"
        if len(seeds) < 5 or independent_layouts < 2
        else "cluster bootstrap; confirmatory status additionally requires declared protocol",
    }
    if comparison:
        differences = draws[:, 0] - draws[:, 1]
        result["difference"] = {
            "methods": list(comparison),
            "estimate": float(estimate[0] - estimate[1]),
            "interval": np.quantile(differences, [alpha, 1 - alpha]).tolist(),
        }
    return result


def pareto_frontier(
    rows: list[dict],
    *,
    minimize: tuple[str, ...] = ("collision", "elapsed_time", "queries"),
    maximize: tuple[str, ...] = ("success",),
) -> list[dict]:
    """Return all nondominated methods, retaining ties rather than arbitrary winners."""
    dimensions = [(key, 1.0) for key in minimize] + [(key, -1.0) for key in maximize]
    valid = [
        row
        for row in rows
        if all(row.get(key) is not None and np.isfinite(float(row[key])) for key, _ in dimensions)
    ]
    values = np.array([[sign * float(row[key]) for key, sign in dimensions] for row in valid])
    return [
        row
        for index, row in enumerate(valid)
        if not any(
            np.all(values[other] <= values[index]) and np.any(values[other] < values[index])
            for other in range(len(valid))
            if other != index
        )
    ]
