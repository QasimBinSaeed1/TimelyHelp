"""Scientific pairing must fail on incomplete traces and unmatching prefixes."""

import pytest

from timelyhelp.counterfactual import CounterfactualRunner
from timelyhelp.simulation import ReferenceSimulator
from timelyhelp.types import Candidate, SimConfig
from timelyhelp import experiments
from timelyhelp.config import load_config


@pytest.mark.parametrize("damage", ["missing_trace", "shifted_time", "prefix_rgb"])
def test_pair_rejects_missing_or_misaligned_evidence(monkeypatch, damage):
    env = ReferenceSimulator(seed=1, config=SimConfig(horizon=0.6))
    runner = CounterfactualRunner(env)
    original = runner._continuation

    def damaged(snapshot, candidate, branch, replication):
        outcome, trace, trial, prefix = original(snapshot, candidate, branch, replication)
        if branch == "Q":
            if damage == "missing_trace":
                trace = []
            elif damage == "shifted_time":
                trace[0]["time"] += 0.01
            else:
                prefix["rgb"][:] = 255 - prefix["rgb"]
        return outcome, trace, trial, prefix

    monkeypatch.setattr(runner, "_continuation", damaged)
    with pytest.raises(AssertionError, match="prearrival"):
        runner.run(env.snapshot(), Candidate("infrastructure", "cautious", 0.3))


def test_native_style_continuation_releases_driver_on_exception(monkeypatch):
    env = ReferenceSimulator(seed=1, config=SimConfig(horizon=0.6))
    snapshot = env.snapshot()
    trial = env.fork(snapshot)
    closed = []

    def fail(*args):
        raise RuntimeError("native capture failed")

    monkeypatch.setattr(trial, "advance", fail)
    trial.close = lambda: closed.append(True)
    runner = CounterfactualRunner(env)
    monkeypatch.setattr(runner, "_spawn", lambda snapshot: trial)
    with pytest.raises(RuntimeError, match="native capture failed"):
        runner._continuation(snapshot, Candidate("none", "hold", 0.2), "W", 0)
    assert closed == [True]


def test_generation_never_retries_failed_scientific_audit(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["simulation"].update(horizon=0.5)
    config["generation"].update(snapshot_times=[0.1], snapshots_per_episode=1)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError("W/Q prearrival failed")

    monkeypatch.setattr(experiments.CounterfactualRunner, "run", fail)
    scenario = experiments._scenario_list(config, "train", 1)[0]
    with pytest.raises(AssertionError, match="prearrival"):
        experiments._collect_episode(config, "local", scenario, 1, audit_restoration=False)
    assert calls == [True]
