"""Actual bounded oracle continuations and retained source imperfections."""

import numpy as np
import pytest

from timelyhelp.audits import audit_public_observation
from timelyhelp.baselines import Decision
from timelyhelp.oracle import run_oracle_episode
from timelyhelp.simulation import ReferenceSimulator, make_scenario
from timelyhelp.types import SimConfig


def test_oracle_executes_real_controlled_labels_and_separate_mission_metrics():
    env = ReferenceSimulator(
        make_scenario(split="test"), seed=2, config=SimConfig(horizon=0.3, max_queries=1)
    )
    result = run_oracle_episode(env, [0.0, 0.1], replications=2, max_decisions=1)
    assert result["steps"] == 3 and not result["success"]
    assert result["elapsed_time"] == pytest.approx(0.3) and result["completion_time"] is None
    assert result["privileged"] and len(result["decisions"]) == 1
    assert result["decisions"][0]["counterfactual_audit"]["passed"]
    assert result["decisions"][0]["counterfactual_audit"]["label_rows"] == 16
    assert result["decisions"][0]["counterfactual_audit"]["A_interactions"] == 2
    assert result["oracle_provenance"]["unknown_and_no_response_preserved"]
    assert not result["oracle_provenance"]["labels_enter_public_observations"]
    assert audit_public_observation(env.observe(), env.history_features())["passed"]
    assert result["queries"] <= 1
    assert len(env.control_log) == 3  # Label continuations did not mutate this mission.


@pytest.mark.parametrize("missing", [False, True])
def test_selected_oracle_time_keeps_unknown_and_missing_actual_source_reply(monkeypatch, missing):
    # Force only the selector's choice to exercise the selected-time execution
    # contract; every engine continuation and source response is still genuine.
    monkeypatch.setattr(
        "timelyhelp.oracle.DiagnosticOracle.select_counterfactuals",
        lambda self, records, **kwargs: Decision("supervisor", "hold", 0.1),
    )
    config = SimConfig(horizon=0.4, request_expiry=0.2, max_queries=1)
    if missing:
        scenario = make_scenario(split="test", disturbances=("no_response",))
        seed = next(
            seed
            for seed in range(100)
            if ReferenceSimulator(scenario, seed, config).draw("no_response", 1) < 0.6
        )
    else:
        scenario, seed = make_scenario(split="test", family="cart_junction"), 2
    env = ReferenceSimulator(scenario, seed, config)
    result = run_oracle_episode(env, [0.0, 0.1])
    assert result["queries"] == 1 and result["request_decisions"][0]["selected_delay"] == 0.1
    assert env.request_log[0]["exact_delay"] == 0.1
    assert env.control_log[0]["linear"] == 0.0
    assert any(control["linear"] > 0 for control in env.control_log[1:])
    if missing:
        assert not np.isfinite(env.pending[0].arrival_time)
        assert result["oracle_expired_requests"] == 1 and not result["reply_delays"]
    else:
        assert result["oracle_reply_status_counts"]["unknown"] == 1
        assert result["reply_delays"] == pytest.approx([0.1])
        assert not env.pending[0].admitted


def test_oracle_decision_budget_and_validation_are_explicit():
    env = ReferenceSimulator(config=SimConfig(horizon=0.3, commitment=0.1, max_queries=0))
    result = run_oracle_episode(env, [0.0, 0.1], max_decisions=2)
    assert len(result["decisions"]) == 2 and result["queries"] == 0
    assert all(row["source"] == "none" for row in result["decisions"])
    assert result["oracle_provenance"]["decision_interval_seconds"] == 0.1
    unlimited = run_oracle_episode(
        ReferenceSimulator(config=SimConfig(horizon=0.3, commitment=0.1, max_queries=0)),
        [0.0],
        max_decisions=None,
    )
    assert len(unlimited["decisions"]) == 3 and unlimited["steps"] == 3
    for options in (
        {"delays": []},
        {"delays": [float("inf")]},
        {"delays": [0.0], "replications": 0},
        {"delays": [0.0], "max_decisions": 0},
        {"delays": [0.0], "decision_interval_s": 0.0},
    ):
        with pytest.raises(ValueError):
            run_oracle_episode(ReferenceSimulator(config=SimConfig(horizon=0.1)), **options)
