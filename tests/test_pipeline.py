"""A genuinely short full-stage run, including fitting and deployed calibration."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from timelyhelp.config import load_config
from timelyhelp.experiments import (
    generate_data,
    train_pipeline,
    evaluate_pipeline,
    report_pipeline,
    _transfer_scenarios,
    _verified_gate,
    _best_mean_no_query_motion,
    _privileged_mission_report,
)
from timelyhelp.learning.data import PairedDataset


def test_transfer_protocol_has_three_physical_layouts():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    scenarios = _transfer_scenarios(config, 100)
    assert len({s.layout_id for s in scenarios}) == 3
    assert {s.variant for s in scenarios} == {0, 1, 2}
    assert {s.family for s in scenarios} == {"worker_crossing", "cart_junction", "ambiguous_intent"}
    assert all(s.setting == "worksite" and s.split == "test" for s in scenarios)


def test_no_query_headroom_compares_means_before_selecting_motion():
    rows = [
        dict(
            source=source,
            bridge="hold",
            delay=1.0,
            replication=replication,
            outcomes={"A": {"cost": 2.0}, "W": {"cost": cost}},
        )
        for source in ("infrastructure", "supervisor")
        for replication, cost in enumerate((0.0, 4.0))
    ]
    # A lucky draw has effect +2, while this motion's mean effect is zero.
    assert _best_mean_no_query_motion(rows) == 0.0
    # Uneven source duplication must not reweight a reused stochastic W control.
    assert _best_mean_no_query_motion(rows + [rows[0]]) == 0.0


def test_changed_checkpoint_is_rejected_before_evaluation(tmp_path):
    checkpoint = tmp_path / "gate.pt"
    selection = tmp_path / "selection.json"
    checkpoint.write_bytes(b"changed weights")
    selection.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint differs"):
        _verified_gate(
            {"model": str(checkpoint), "selection": str(selection), "weight_sha256": "old checksum"}
        )
    with pytest.raises(ValueError, match="selection thresholds differ"):
        _verified_gate(
            {
                "model": str(checkpoint),
                "selection": str(selection),
                "weight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "selection_sha256": "old selection checksum",
            }
        )


def test_oracle_subset_frontier_uses_exactly_matched_control_missions():
    common = dict(
        backend="reference",
        controller="local",
        training_seed=17,
        layout_id="layout",
        episode_id="matched",
        elapsed_time=1.0,
        collision=False,
        queries=0,
        success=True,
        record_type="mission",
    )
    controls = [
        common | {"method": method} for method in ("nominal", "cautious", "model_based_voi")
    ]
    controls += [
        dict(row, layout_id="other", episode_id="unmatched", success=False) for row in controls
    ]
    oracle = [
        common
        | {"method": "privileged_oracle", "record_type": "privileged_mission", "privileged": True}
    ]
    report = _privileged_mission_report(controls, oracle, samples=10, seed=0)
    cohort = report["cohorts"]["reference/local"]
    assert all(row["missions"] == 1 and row["success"] == 1.0 for row in cohort["summary"])
    assert cohort["paired_comparisons"]["model_based_voi"]["success"]["matched_missions"] == 1


def test_complete_bounded_pipeline_and_resume_artifacts(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["simulation"].update(horizon=0.4, max_queries=1, commitment=0.2)
    config["generation"].update(
        episodes={split: 1 for split in ("train", "development", "calibration", "test")},
        snapshot_times=[0.1],
        delays=[0.0, 0.2],
    )
    config["learning"].update(hidden_dim=8, epochs=1, ensemble_size=2)
    config["evaluation"].update(episodes=1, bootstrap_samples=10)
    config["controllers"] = ["local"]
    data = generate_data(config, tmp_path / "data")
    assert data["raw_rows"] == 32
    dataset = PairedDataset.load(data["dataset"])
    assert {m["split"] for m in dataset.metadata} == {"train", "development", "calibration", "test"}
    assert all(
        audit["passed"]
        for audit in json.loads((tmp_path / "data/restoration_audits.json").read_text())
    )
    models = train_pipeline(config, data, tmp_path / "models")
    fits = models["controllers"]["local"]["seeds"]["17"]
    assert set(fits) == {"timelyhelp", *config["ablations"]}
    selection = json.loads(Path(fits["timelyhelp"]["selection"]).read_text())
    assert selection["evaluation"] == "closed_loop" and not selection["calibration_feasible"]
    assert selection["actual_collision_upper_95"] > 0.1
    raw_labels = json.loads(Path(data["records"]).read_text())
    residual_calibration_seeds = {r["seed"] for r in raw_labels if r["split"] == "calibration"}
    assert set(selection["rollout_seeds"]).isdisjoint(residual_calibration_seeds)
    assert selection["rollout_seeds"] == [config["scenario_seed"] + 40000]
    evaluation = evaluate_pipeline(config, models, tmp_path / "evaluation", dataset=data)
    records = json.loads(Path(evaluation["records"]).read_text())
    assert len(records) == 15
    assert {record["method"] for record in records} >= set(config["evaluation"]["methods"])
    failure = next(record for record in records if record["method"] == "failure_lead_time")
    assert (
        failure["failure_predictor_provenance"]["weight_sha256"]
        == fits["timelyhelp"]["weight_sha256"]
    )
    assert failure["failure_predictor_provenance"]["no_additional_training"]
    assert (
        failure["decisions"][0]["candidate_scores"][0]["probability_semantics"]
        == "calibrated nominal A mission-failure forecast"
    )
    assert evaluation["distinct_oracle_missions_executed"] == 1
    privileged = json.loads(Path(evaluation["privileged_oracle_records"]).read_text())
    assert len(privileged) == 1 and privileged[0]["record_type"] == "privileged_mission"
    assert privileged[0]["privileged"] and privileged[0]["method"] == "privileged_oracle"
    assert privileged[0]["episode_id"] == records[0]["episode_id"]
    assert all(not r.get("privileged", False) for r in records)
    asks = [record for record in records if record["queries"] > 0]
    assert asks and all(record["useless_requests_audited"] is not None for record in asks)
    report = report_pipeline(config, evaluation, tmp_path / "report", data, models)
    assert len(report["figures"]) == 6
    assert Path(report["oracle_headroom"]).is_file()
    oracle_diagnostic = json.loads(Path(report["oracle_mission_diagnostics"]).read_text())
    comparisons = oracle_diagnostic["cohorts"]["reference/local"]["paired_comparisons"]
    assert {"nominal", "cautious", "model_based_voi"} <= set(comparisons)
    assert comparisons["model_based_voi"]["success"]["matched_missions"] == 1
    assert json.loads((tmp_path / "report/window_predictions.json").read_text())
    assert all(np.isfinite(record["elapsed_time"]) for record in records)
