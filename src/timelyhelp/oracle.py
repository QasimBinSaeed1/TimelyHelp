"""Privileged source/time oracle missions, separated from deployable policies.

This diagnostic obtains complete-state controlled continuations and can choose a
response time from the declared probe grid. It does not repair a source's payload,
force an answer to exist, or bypass the shared evidence/safety path. Finite Monte
Carlo selection and repeated decisions do not establish a global optimality bound.
"""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Sequence

import numpy as np

from .audits import audit_controlled_labels
from .baselines import DiagnosticOracle
from .counterfactual import CounterfactualRunner
from .types import Candidate


def run_oracle_episode(
    env,
    delays: Sequence[float],
    replications: int = 1,
    max_decisions: int | None = 1,
    decision_interval_s: float | None = None,
) -> dict:
    """Execute a distinctly privileged, bounded source/time diagnostic mission.

    Every decision scores actual A/W/Q continuations of this engine with shared
    Monte Carlo draws. No-query nominal and reusable cautious/holding controls
    compete with queries. Query times are ideal interventions selected from the
    delay grid; actual imperfect, unknown, missing and stale responses remain.

    At a subsequent decision, pending queries are cancelled before constructing
    matched continuations. The selected bridge lasts until its selected response
    time unless the next finite decision supersedes it. Once the decision budget
    is exhausted, the bridge finishes and nominal control completes the mission.
    No privileged labels are inserted into ordinary observations or features.
    """
    grid = sorted(set(float(delay) for delay in delays))
    if not grid or not np.isfinite(grid).all() or min(grid) < 0:
        raise ValueError("Oracle delay probes must be finite and nonnegative")
    for name, value in (("replications", replications), ("max_decisions", max_decisions)):
        if name == "max_decisions" and value is None:
            continue  # Research configuration: repeat finite commitments until mission termination.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Oracle {name} must be a positive integer")
    interval = env.config.commitment if decision_interval_s is None else float(decision_interval_s)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Oracle decision interval must be finite and positive")
    # A live learned admission closure would make counterfactual outcomes depend
    # on another selector and could capture an engine outside these branches.
    if env.admission_filter is not None:
        raise ValueError(
            "Privileged oracle requires a fresh environment with common evidence admission"
        )

    selector = DiagnosticOracle()
    decisions, request_decisions, selection_seconds = [], [], []
    next_decision = float(env.time)
    bridge, bridge_until = "nominal", float(env.time)
    steps = 0
    maximum_steps = int(np.ceil(env.config.horizon / env.config.dt)) + 2
    while not env.done and steps < maximum_steps:
        if (
            max_decisions is None or len(decisions) < max_decisions
        ) and env.time >= next_decision - 1e-9:
            cancelled = []
            for pending in env.observe().get("pending", []):
                if pending.get("reason") == "pending" and env.cancel(pending["request_id"]):
                    cancelled.append(pending["request_id"])
            snapshot = env.snapshot()
            start = time.perf_counter()
            runner = CounterfactualRunner(env)
            sources = (
                ("infrastructure", "supervisor")
                if env.queries < env.config.max_queries
                else ("none",)
            )
            labels = [
                runner.run(snapshot, Candidate(source, motion, delay), replication)
                for source in sources
                for motion in ("cautious", "hold")
                for delay in grid
                for replication in range(replications)
            ]
            label_audit = audit_controlled_labels(labels)
            decision = selector.select_counterfactuals(
                labels, query_cost=env.config.query_cost, risk_limit=1.0
            )
            selection_seconds.append(time.perf_counter() - start)
            bridge = decision.bridge
            bridge_until = float(env.time) + decision.delay
            request_id = None
            if decision.source != "none":
                request_id = env.request(
                    Candidate(decision.source, decision.bridge, decision.delay),
                    exact_delay=decision.delay,
                )
                request_decisions.append(
                    {
                        "time": float(env.time),
                        "source": decision.source,
                        "bridge": decision.bridge,
                        "request_id": request_id,
                        "selected_delay": decision.delay,
                        "forced_arrival_delay": True,
                    }
                )
            decisions.append(
                {
                    **asdict(decision),
                    "time": float(snapshot["time"]),
                    "request_id": request_id,
                    "cancelled_request_ids": cancelled,
                    "counterfactual_audit": label_audit,
                    "privileged": True,
                }
            )
            next_decision = float(snapshot["time"]) + interval
        env.advance(bridge if env.time < bridge_until - 1e-9 else "nominal")
        steps += 1

    outcome = dict(env.outcomes())
    elapsed = float(outcome["time"])
    completed, collision = bool(outcome["completion"]), bool(outcome["collision"])
    within_budget = elapsed <= env.config.horizon + env.config.dt * 1e-6
    truncated = not env.done
    observation = env.observe()
    replies = observation.get("messages", [])
    pending = observation.get("pending", [])
    reply_delays = outcome.get("reply_delays", [])
    outcome.update(
        completed=completed,
        success=bool(completed and not collision and within_budget and not truncated),
        completion_time=elapsed if completed else None,
        elapsed_time=elapsed,
        within_budget=within_budget,
        truncated=truncated,
        steps=steps,
        reply_delay=float(np.mean(reply_delays)) if reply_delays else None,
        supervisor_workload=outcome.get("workload"),
        late_requests=sum(
            message.get("reason") in ("stale", "expired", "newer_belief", "newer_source_belief")
            for message in replies
        )
        + sum(request.get("reason") == "expired" for request in pending),
        useless_requests_audited=None,
        deadline_error=None,
        decisions=decisions,
        request_decisions=request_decisions,
        selection_seconds_mean=float(np.mean(selection_seconds)) if selection_seconds else 0.0,
        selection_seconds_max=max(selection_seconds, default=0.0),
        privileged=True,
        oracle_provenance={
            "role": "separately reported privileged source/time mission diagnostic",
            "privilege": "complete-state initialization, full controlled continuation outcomes, and ideal selected arrival-delay interventions",
            "delay_probes": grid,
            "monte_carlo_replications": replications,
            "maximum_decisions": max_decisions,
            "decision_interval_seconds": interval,
            "query_cost": env.config.query_cost,
            "unknown_and_no_response_preserved": True,
            "shared_evidence_and_safety": True,
            "labels_enter_public_observations": False,
            "draws": "shared seed/replication/tick/channel draws within each A/W/Q comparison",
            "limitation": "Finite Monte Carlo and replanning are diagnostic; no global optimality or collision guarantee. Ideal selected latency is not deployable source scheduling.",
        },
        oracle_reply_status_counts={
            reason: sum(message.get("reason") == reason for message in replies)
            for reason in ("admitted", "unknown", "stale", "expired")
        },
        oracle_expired_requests=sum(request.get("reason") == "expired" for request in pending),
        oracle_cancelled_requests=sum(request.get("reason") == "cancelled" for request in pending),
    )
    return outcome
