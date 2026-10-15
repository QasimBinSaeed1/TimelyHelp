from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest

from timelyhelp.audits import (
    audit_controlled_labels,
    audit_public_observation,
    matched_observable_scenes,
    monte_carlo_summary,
    pilot_diagnostics,
    process_tree_memory,
    profile_resources,
    window_prediction_grid,
)
from timelyhelp.counterfactual import CounterfactualRunner
from timelyhelp.learning import Candidate, ConsequenceEnsemble, ModelConfig, PairedDataset
from timelyhelp.simulation import ReferenceSimulator, make_scenario
from timelyhelp.types import SimConfig


def paired_rows():
    env = ReferenceSimulator(
        make_scenario(split="train"),
        seed=11,
        config=SimConfig(horizon=2.0, camera_width=24, camera_height=6),
    )
    runner = CounterfactualRunner(env)
    snapshot = env.snapshot()
    return [
        runner.run(snapshot, Candidate(source, bridge, delay), replication)
        for source in ("infrastructure", "supervisor")
        for bridge in ("cautious", "hold")
        for delay in (0.0, 0.5)
        for replication in range(2)
    ]


def test_real_sensor_observation_schema_and_future_payload_rejection():
    env = ReferenceSimulator(make_scenario(split="train"), config=SimConfig(horizon=2.0))
    assert audit_public_observation(env.observe(), env.history_features())["passed"]
    poisoned = deepcopy(env.observe())
    poisoned["true_reliability"] = 0.9
    with pytest.raises(ValueError, match="Privileged"):
        audit_public_observation(poisoned)
    poisoned = deepcopy(env.observe())
    poisoned["pending"] = [
        {
            "request_id": 1,
            "source": "supervisor",
            "reason": "pending",
            "cancelled": False,
            "expires_at": 2.0,
            "arrival_time": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="Nonpublic"):
        audit_public_observation(poisoned)
    poisoned = deepcopy(env.observe())
    poisoned["messages"] = [
        dict(
            request_id=1,
            source="supervisor",
            question="will_you_cross_delivery_lane",
            capture_time=0.0,
            arrival_time=1.0,
            payload={"answer": "unknown"},
            admitted=False,
            reason="unknown",
        )
    ]
    with pytest.raises(ValueError, match="Unarrived"):
        audit_public_observation(poisoned)


def test_controlled_accounting_unique_reuse_and_full_failure_retention():
    rows = paired_rows()
    report = audit_controlled_labels(rows)
    assert report["prearrival_status"] == "verified"
    assert report["A_interactions"] == 2
    assert report["W_interactions"] == 8
    assert report["Q_interactions"] == 16
    assert report["distinct_branch_interactions"] == 26
    assert report["failures_retained"] > 0  # short horizon intentionally fails
    assert sum(report["signed_rows"].values()) == len(rows)
    damaged = deepcopy(rows)
    damaged[0]["info"] += 1
    with pytest.raises(ValueError, match="decomposition"):
        audit_controlled_labels(damaged)
    damaged = deepcopy(rows)
    damaged[0]["prearrival_audit"]["passed"] = False
    with pytest.raises(ValueError, match="Prearrival"):
        audit_controlled_labels(damaged)


def test_MC_oracle_and_window_grid_have_diagnostic_provenance():
    rows = paired_rows()
    summaries = monte_carlo_summary(rows)
    assert len(summaries) == 8 and all(row["replications"] == 2 for row in summaries)
    assert all(row["sampling_uncertainty_identified"] for row in summaries)
    one = [row for row in rows if row["replication"] == 0]
    assert all(row["effect_standard_error"] is None for row in monte_carlo_summary(one))
    pilot = pilot_diagnostics(rows)
    assert "privileged" in pilot["role"]
    assert pilot["snapshot_count"] == 1
    assert pilot["oracle_mean_gap_over_strong_noquery"] >= 0
    assert not pilot["central_hypothesis_supported_by_this_pilot"]
    dataset = PairedDataset.from_records(rows)
    model = ConsequenceEnsemble(ModelConfig(epochs=1, ensemble_size=1, hidden_dim=8))
    model.fit(dataset)
    windows = window_prediction_grid(model, dataset, [0.0, 0.5, 1.0])
    assert len(windows) == 4 and all(row["probe_horizon"] == 1.0 for row in windows)
    assert all("right_censored" in row for row in windows)


def test_observable_pair_search_explains_aliases_and_metadata_differences():
    first = paired_rows()
    # This diagnostic deliberately creates observational aliasing with opposite
    # oracle choices. It must identify unpredictability instead of claiming that
    # hidden realized outcomes permit a deployable distinction.
    second = deepcopy(first)
    for record in first:
        record["outcomes"]["Q"]["cost"] = 0.0 if record["source"] == "infrastructure" else 1.0
        record["info"] = record["outcomes"]["W"]["cost"] - record["outcomes"]["Q"]["cost"]
    for record in second:
        record["snapshot_id"] = "aliased-other-snapshot"
        record["rollout_id"] = "other-rollout"
        record["outcomes"]["Q"]["cost"] = 0.0 if record["source"] == "supervisor" else 1.0
        record["info"] = record["outcomes"]["W"]["cost"] - record["outcomes"]["Q"]["cost"]
    aliases = matched_observable_scenes(first + second)
    assert aliases["identical_observation_aliases"]
    assert "not predictably" in aliases["identical_observation_aliases"][0]["interpretation"]
    for record in second:
        h = np.asarray(record["history"])
        h[:, 18] = 1.0
        record["history"] = h.tolist()
    explained = matched_observable_scenes(first + second)
    assert explained["identical_confidence_pairs"]
    assert (
        explained["identical_confidence_pairs"][0]["accessible_differences"][0]["feature"]
        == "infrastructure_delay_mean"
    )


def test_resource_profile_measures_steps_and_keeps_original_state():
    env = ReferenceSimulator(make_scenario(split="train"), config=SimConfig(horizon=2.0))
    before = env.pose.copy()
    report = profile_resources(env, steps=2)
    np.testing.assert_array_equal(before, env.pose)
    assert env.time == 0
    assert report["steps"] == 2 and report["simulation_seconds"] > 0
    assert report["backend"] == "reference" and not report["native_simulator_profiled"]
    assert report["model_latency_seconds_mean"] is None
    assert not report["native_worker_memory_before"]["available"]
    assert report["native_worker_memory_after"]["worker_pid"] is None
    assert "all jobs" in report["gpu_device_memory_scope"]


def test_process_tree_memory_measures_only_owned_processes_and_descendants():
    script = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
def terminate(signum, frame):
    raise SystemExit
signal.signal(signal.SIGTERM, terminate)
print(child.pid, flush=True)
try:
    time.sleep(10)
finally:
    child.terminate()
    child.wait()
"""
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(process.stdout.readline())
        measured = process_tree_memory(process.pid)
        assert measured["available"] and measured["sampled_tree_rss_bytes"] > 0
        by_pid = {entry["pid"]: entry for entry in measured["processes"]}
        assert set(by_pid) == {process.pid, child_pid}
        assert by_pid[process.pid]["rss_bytes"] > 0
        assert by_pid[child_pid]["parent_pid"] == process.pid
        assert measured["sum_process_high_water_rss_bytes"] >= measured["sampled_tree_rss_bytes"]
        assert "not a simultaneous" in measured["scope"]
        assert not process_tree_memory(999999999)["available"]
    finally:
        process.terminate()
        process.wait(timeout=2)
        process.stdout.close()
