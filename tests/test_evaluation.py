"""Statistical-unit, scheduling and observation-only comparator integration tests."""

from dataclasses import dataclass
import json

import numpy as np
import pytest

from timelyhelp.baselines import (
    MAIN_BASELINES,
    BaselineConfig,
    DiagnosticOracle,
    calibrate_source_metadata,
    make_baseline,
)
from timelyhelp.evaluation import (
    hierarchical_bootstrap,
    pareto_frontier,
    run_episode,
    run_evaluation,
)
from timelyhelp.reporting import (
    generate_report,
    sampled_windows,
    window_diagnostics,
    frozen_gate_audit,
)
from timelyhelp.simulation.reference import ReferenceSimulator
from timelyhelp.simulation.scenarios import make_scenario
from timelyhelp.types import SimConfig


def _rows():
    return [
        {
            "method": method,
            "record_type": "mission",
            "training_seed": seed,
            "layout_id": f"layout{layout}",
            "episode_id": f"{layout}:{episode}",
            "success": float((seed + layout + episode) % 2) + offset,
            "queries": 2.0 - offset,
        }
        for seed in range(5)
        for layout in range(2)
        for episode in range(3)
        for method, offset in (("first", 0.25), ("second", 0.0))
    ]


def test_bootstrap_keeps_matched_subjects_and_seed_hierarchy():
    result = hierarchical_bootstrap(_rows(), "success", comparison=("first", "second"), samples=80)
    assert result["independent_training_seeds"] == 5
    assert result["layouts"] == 2
    assert result["matched_missions"] == 30
    assert result["difference"]["estimate"] == pytest.approx(0.25)
    assert result["difference"]["interval"] == pytest.approx([0.25, 0.25])


def test_bootstrap_requires_distinct_backend_cohorts():
    rows = [dict(row, backend="isaac") for row in _rows()]
    rows += [dict(row, backend="gazebo") for row in _rows()]
    with pytest.raises(ValueError, match="one backend"):
        hierarchical_bootstrap(rows, "success", samples=10)


def test_report_keeps_native_backend_missions_separate(tmp_path):
    rows = [
        dict(
            row,
            backend=backend,
            collision=0.0,
            elapsed_time=1.0,
            completed=True,
            completion_time=1.0,
            frozen_gate=backend == "gazebo",
        )
        for backend in ("isaac", "gazebo")
        for row in _rows()
    ]
    generate_report(
        rows, [], tmp_path, config={"frozen_gate_sha256": "recorded_hash"}, bootstrap_samples=10
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["confidence_intervals_by_backend"]) == {"isaac", "gazebo"}
    assert report["confidence_intervals_by_backend"]["isaac"]["success"]["matched_missions"] == 30
    assert report["confidence_intervals_by_backend"]["gazebo"]["success"]["matched_missions"] == 30
    assert len(report["summary"]) == 4
    assert {r["backend"] for r in report["summary"]} == {"isaac", "gazebo"}


def test_native_transfer_figure_requires_matching_gate_identity():
    common = dict(
        method="timelyhelp",
        controller="local",
        training_seed=17,
        gate_weight_sha256="weights",
        gate_selection_sha256="thresholds",
    )
    primary = common | {"backend": "isaac", "frozen_gate": False}
    transfer = common | {"backend": "gazebo", "frozen_gate": True}
    assert frozen_gate_audit([primary, transfer])["verified"]
    changed = transfer | {"gate_weight_sha256": "other weights"}
    assert not frozen_gate_audit([primary, changed])["verified"]
    changed = transfer | {"gate_selection_sha256": "other thresholds"}
    assert not frozen_gate_audit([primary, changed])["verified"]


def test_window_error_uses_selected_seed_risk_threshold():
    common = {
        "layout": "test",
        "rollout_id": "rollout",
        "snapshot_id": "snapshot",
        "source": "infrastructure",
        "bridge": "cautious",
    }
    rows = [
        common | {"delay": delay, "info": info, "outcomes": {"Q": {"collision": 0.2}}}
        for delay, info in ((0.0, 1.0), (1.0, 1.0), (2.0, -1.0))
    ]
    predictions = [
        common
        | {
            "delay": delay,
            "training_seed": 17,
            "predicted_info": info,
            "predicted_collision": 0.2,
            "risk_limit": 0.4,
        }
        for delay, info in ((0.0, 1.0), (1.0, 1.0), (2.0, -1.0))
    ]
    result = window_diagnostics(rows, predictions, risk_limit=0.1)[0]
    assert result["actual"]["intervals"] == []
    assert result["predictions_by_training_seed"][0]["actual_at_selected_risk_limit"][
        "intervals"
    ] == [[0.0, 1.0]]
    assert result["deadline_absolute_error"] == 0.0
    assert result["grid_classification_error"] == 0.0


def test_sibling_branches_and_duplicates_rejected():
    rows = _rows()
    with pytest.raises(ValueError, match="siblings"):
        hierarchical_bootstrap([dict(rows[0], record_type="counterfactual")], "success")
    with pytest.raises(ValueError, match="Duplicate"):
        hierarchical_bootstrap(rows + [rows[0]], "success")


def test_frontier_retains_equal_points_and_tradeoffs():
    rows = [
        dict(method="safe", collision=0.0, elapsed_time=5.0, queries=2.0, success=1.0),
        dict(method="fast", collision=0.1, elapsed_time=3.0, queries=1.0, success=0.9),
        dict(method="dominated", collision=0.2, elapsed_time=6.0, queries=3.0, success=0.8),
        dict(method="safe_duplicate", collision=0.0, elapsed_time=5.0, queries=2.0, success=1.0),
    ]
    assert {r["method"] for r in pareto_frontier(rows)} == {"safe", "fast", "safe_duplicate"}


def test_disconnected_and_censored_windows_are_not_forced_to_deadlines():
    result = sampled_windows([0, 0.25, 0.75, 1.5, 3], [1, 1, -1, 1, 1], [0] * 5)
    assert result["intervals"] == [[0.0, 0.25], [1.5, 3.0]]
    assert result["right_censored"] and result["disconnected"]
    assert result["connected_deadline"] is None
    assert sampled_windows([0, 1, 2], [1, 1, -1], [0] * 3)["connected_deadline"] == 1.0


def test_all_main_rules_and_model_based_voi_run_public_pipeline():
    config = SimConfig(horizon=0.3, max_queries=1)
    env = ReferenceSimulator(make_scenario(), seed=3, config=config)
    for name in MAIN_BASELINES:
        policy = make_baseline(
            name, BaselineConfig(sim=config, monte_carlo_samples=2, latency_support=1)
        )
        observation = env.observe()
        metadata = observation["sources"]
        decision = policy.select(env.history_features(), metadata, observation=observation)
        assert decision.bridge in ("nominal", "cautious", "hold")
        assert decision.source in ("none", "infrastructure", "supervisor")
        assert np.isfinite(decision.net_value)
    result = run_episode(
        ReferenceSimulator(make_scenario(), seed=3, config=config),
        make_baseline(
            "model_based_voi", BaselineConfig(sim=config, monte_carlo_samples=2, latency_support=1)
        ),
    )
    assert result["steps"] == 3 and not result["success"]
    assert result["queries"] <= 1


def test_failure_lead_time_uses_calibrated_predictions_not_heuristic():
    class Predictor:
        calibration = {"split": "calibration"}

        def __init__(self, completion):
            self.completion = completion
            self.observed = None

        def predict(self, history, candidates):
            self.observed = np.array(history, copy=True)
            assert len(candidates) == 1 and candidates[0].source == "none"
            return {
                "collision": np.array([[0.01, 0.01, 0.01]]),
                "completion": np.full((1, 3), self.completion),
                "collision_uncertainty": np.zeros((1, 3)),
                "completion_uncertainty": np.zeros((1, 3)),
            }

    env = ReferenceSimulator(config=SimConfig(horizon=12.0))
    observation = env.observe()
    observation["blind_zone_uncertainty"] = 1.0  # Heuristic monitor score would be high.
    history = env.history_features()
    low, high = Predictor(0.95), Predictor(0.1)
    config = BaselineConfig(sim=env.config, failure_threshold=0.2)
    conservative = make_baseline("failure_lead_time", config, failure_predictor=low)
    anticipating = make_baseline("failure_lead_time", config, failure_predictor=high)
    first = conservative.select(history, observation["sources"], observation)
    second = anticipating.select(history, observation["sources"], observation)
    assert first.source == "none" and second.source != "none"
    assert first.candidate_scores[0]["failure_probability"] == pytest.approx(0.05)
    assert second.candidate_scores[0]["failure_probability"] == pytest.approx(0.9)
    assert np.array_equal(low.observed, history) and np.array_equal(high.observed, history)
    fallback = make_baseline("failure_lead_time", config)
    assert fallback.name == "failure_monitor_score"
    high.calibration = None
    with pytest.raises(ValueError, match="calibrated predictor"):
        make_baseline("failure_lead_time", config, failure_predictor=high)


def test_model_based_voi_inputs_are_only_observable_and_deterministic():
    config = SimConfig(horizon=0.3)
    first = ReferenceSimulator(seed=1, config=config)
    second = first.fork()
    # Mutating hidden truth without sensing cannot influence the policy input.
    second.hidden_source_error = {"infrastructure": 0.99, "supervisor": 0.99}
    second.actors[0]["destination"] = np.array([100.0, 100.0])
    observation1, observation2 = first.observe(), second.observe()
    assert observation1 == observation2
    policy1 = make_baseline(
        "model_based_voi",
        BaselineConfig(sim=config, monte_carlo_samples=2, latency_support=1),
        seed=17,
    )
    policy2 = make_baseline(
        "model_based_voi",
        BaselineConfig(sim=config, monte_carlo_samples=2, latency_support=1),
        seed=17,
    )
    one = policy1.select(first.history_features(), observation1["sources"], observation1)
    two = policy2.select(second.history_features(), observation2["sources"], observation2)
    assert one == two


def test_belief_objective_uses_absolute_mission_time():
    from timelyhelp.baselines import _BeliefRollout

    config = SimConfig(horizon=2.0, dt=0.1)
    observation = ReferenceSimulator(config=config).observe()
    observation.update(time=1.5, remaining_time=0.5, goal=[100.0, 0.0], map_obstacles=[], tracks=[])
    belief = _BeliefRollout(observation, [], config, np.zeros((5, 8)))
    cost, collision, completed, elapsed = belief.run("hold", 0.5, None)
    assert elapsed == pytest.approx(0.5)
    assert not collision and not completed
    assert cost == pytest.approx(config.failure_weight + config.time_weight)


@pytest.mark.parametrize("response_fraction,delay", [(0.0, 0.1), (1.0, 0.7)])
def test_voi_absent_or_expired_reply_uses_finite_expiration_bridge(
    monkeypatch, response_fraction, delay
):
    from timelyhelp.baselines import _BeliefRollout

    config = SimConfig(horizon=0.3, request_expiry=0.2, commitment=0.1)
    env = ReferenceSimulator(config=config)
    observation = env.observe()
    metadata = {
        "infrastructure": dict(
            observation["sources"]["infrastructure"],
            latency_samples=[delay],
            response_fraction=response_fraction,
        )
    }
    durations = []
    original = _BeliefRollout.run

    def traced(self, bridge, duration, packet):
        durations.append((bridge, duration, packet is not None))
        return original(self, bridge, duration, packet)

    monkeypatch.setattr(_BeliefRollout, "run", traced)
    policy = make_baseline(
        "model_based_voi", BaselineConfig(sim=config, monte_carlo_samples=2, latency_support=1)
    )
    decision = policy.select(env.history_features(), metadata, observation)
    assert decision.source == "none"
    assert any(duration == 0.2 for _, duration, _ in durations)
    assert all(duration <= 0.2 for _, duration, _ in durations)
    assert all(not packet for _, _, packet in durations)
    assert all(
        score["info"] == 0.0 for score in decision.candidate_scores if score["source"] != "none"
    )


def test_runtime_latency_is_source_sample_not_candidate_prediction():
    @dataclass
    class Ask:
        source: str = "infrastructure"
        bridge: str = "hold"
        delay: float = 0.0

    class Policy:
        def select(self, history, metadata):
            return Ask()

    config = SimConfig(
        horizon=0.8,
        max_queries=2,
        commitment=0.2,
        source_latency={"infrastructure": [0.2], "supervisor": [0.2]},
    )
    env = ReferenceSimulator(seed=7, config=config)
    result = run_episode(env, Policy())
    assert result["reply_delays"] and min(result["reply_delays"]) >= 0.3 - 1e-8
    assert all(
        event.get("exact_delay") is None
        for event in env.request_log
        if event["action"] == "request"
    )
    assert result["queries"] <= 2


def test_cancellation_and_finite_commitment():
    @dataclass
    class Choice:
        source: str
        bridge: str = "hold"
        delay: float = 2.0

    class Policy:
        def select(self, history, metadata, observation=None):
            return Choice(
                "supervisor" if observation["time"] == 0 else "none",
                "hold" if observation["time"] == 0 else "cautious",
            )

    env = ReferenceSimulator(config=SimConfig(horizon=0.6, commitment=0.2))
    run_episode(env, Policy())
    assert any(event["action"] == "cancel" for event in env.request_log)
    assert env.control_log[0]["linear"] == 0
    assert env.control_log[-1]["linear"] > 0


def test_development_source_calibration_never_uses_test_truth():
    result = calibrate_source_metadata(
        [{"split": "development", "source": "infrastructure", "correct": True}] * 3
    )
    assert result["infrastructure"]["reliability_prior"] == 0.8
    assert result["supervisor"]["calibration_status"] == "declared prior only"
    with pytest.raises(ValueError, match="development"):
        calibrate_source_metadata([{"split": "test", "source": "supervisor", "correct": True}])


def test_oracle_is_separate_from_public_scheduler_and_honors_motion_control():
    oracle = DiagnosticOracle()
    assert not hasattr(oracle, "select")
    row = {
        "source": "supervisor",
        "bridge": "hold",
        "delay": 1.0,
        "outcomes": {
            "A": {"cost": 1.0, "collision": False},
            "W": {"cost": 0.5, "collision": False},
            "Q": {"cost": 0.5, "collision": False},
        },
    }
    assert oracle.select_counterfactuals([row]).source == "none"


def test_evaluation_and_six_report_artifacts_are_generated(tmp_path):
    config = SimConfig(horizon=0.2, max_queries=1)
    records = run_evaluation(
        lambda scenario, seed: ReferenceSimulator(scenario, seed, config),
        {"no_query_nominal": make_baseline("no_query_nominal")},
        [make_scenario("test")],
        seeds=[0],
    )
    assert records[0]["training_seed"] == 0 and records[0]["record_type"] == "mission"
    artifacts = generate_report(records, [], tmp_path, bootstrap_samples=20)
    assert len(artifacts["figures"]) == 6
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["confidence_intervals"]["success"]["independent_training_seeds"] == 1
    assert report["accessible_scene_pairs"] == []
    assert report["provenance"]["backend_counts"] == {"reference": 1}
