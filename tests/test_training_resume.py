"""Fit recovery preserves scientific identity and never reuses stale weights."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pytest

from timelyhelp import experiments
from timelyhelp.config import load_config, write_json
from timelyhelp.learning.data import PairedDataset
from timelyhelp.learning.scheduler import SchedulerConfig
from timelyhelp.types import Candidate


def _small_data():
    splits = ("train", "development", "calibration", "test")
    histories = np.zeros((4, 2, 61), dtype=np.float32)
    histories[:, :, 0] = 0.25
    return PairedDataset(
        histories,
        [Candidate("infrastructure", "cautious", 0.0) for _ in splits],
        np.asarray([[0, 0, 0, 0, 0, 1, 1, 1, 0.4, 0.4, 0.4]] * 4),
        [dict(layout=split, rollout_id=split, snapshot_id=split, split=split) for split in splits],
        np.zeros((4, 11)),
        {"version": 3, "controller": "local"},
    )


def test_fit_resumes_after_threshold_interruption_without_refitting(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["controllers"] = ["local"]
    config["ablations"] = []
    config["simulation"].update(horizon=0.4)
    config["learning"].update(epochs=1, ensemble_size=1, hidden_dim=4)
    data = _small_data()
    calls = {"fit": 0, "threshold": 0}
    original_fit = experiments.ConsequenceEnsemble.fit

    def fit(model, *args, **kwargs):
        calls["fit"] += 1
        return original_fit(model, *args, **kwargs)

    def thresholds(*args, **kwargs):
        calls["threshold"] += 1
        if calls["threshold"] == 1:
            raise RuntimeError("interrupted independent threshold calibration")
        return {
            "split": "calibration",
            "evaluation": "closed_loop",
            "episodes": 1,
            "thresholds": {"risk_limit": 0.1},
            "selected_configuration": asdict(SchedulerConfig()),
        }

    def profile(*args, model_path, **kwargs):
        write_json(Path(model_path).with_suffix(".profile.json"), {"test_profile": True})

    monkeypatch.setattr(experiments.ConsequenceEnsemble, "fit", fit)
    monkeypatch.setattr(experiments, "_threshold_calibration", thresholds)
    monkeypatch.setattr(experiments, "_write_profile", profile)
    output = tmp_path / "models"
    with pytest.raises(RuntimeError, match="interrupted independent"):
        experiments.train_pipeline(config, data, output)
    path = output / "local/seed_17/timelyhelp.pt"
    assert path.with_suffix(".fit.stamp.json").is_file()
    assert not path.with_suffix(".stamp.json").exists()
    original_bytes = path.read_bytes()
    completed = experiments.train_pipeline(config, data, output)
    assert path.read_bytes() == original_bytes
    assert calls == {"fit": 1, "threshold": 2}
    assert (
        completed["controllers"]["local"]["seeds"]["17"]["timelyhelp"]["weight_sha256"]
        == hashlib.sha256(original_bytes).hexdigest()
    )
    experiments.train_pipeline(config, data, output)
    assert calls == {"fit": 1, "threshold": 2}


@pytest.mark.parametrize(
    "change",
    ["missing_stamp", "weights", "fit_report", "config", "dataset", "code", "source", "seed"],
)
def test_unverified_fit_is_rejected_without_overwriting_artifacts(tmp_path, change):
    path = tmp_path / "gate.pt"
    path.write_bytes(b"fitted checkpoint")
    write_json(path.with_suffix(".fit.json"), {"fit": [1], "residual_calibration": [2]})
    identity = experiments._fit_identity({}, "dataset", "timelyhelp", "local", 11, {})
    experiments._write_fit_stamp(path, identity)
    assert experiments._validated_fit_checkpoint(path, identity)
    requested = deepcopy(identity)
    if change == "missing_stamp":
        path.with_suffix(".fit.stamp.json").unlink()
    elif change == "weights":
        path.write_bytes(b"changed checkpoint")
    elif change == "fit_report":
        write_json(path.with_suffix(".fit.json"), {"changed": True})
    else:
        key = {
            "config": "config_sha256",
            "dataset": "dataset_sha256",
            "code": "code_sha256",
            "source": "source_calibration_sha256",
            "seed": "training_seed",
        }[change]
        requested[key] = "changed"
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="Unverified or incompatible fitted checkpoint"):
        experiments._validated_fit_checkpoint(path, requested)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_in_memory_dataset_identity_includes_actual_training_arrays():
    original = _small_data()
    changed = deepcopy(original)
    changed.targets[0, 0] = 1.0
    assert experiments._dataset_identity(original) != experiments._dataset_identity(changed)
    changed = deepcopy(original)
    changed.histories[0, 0, 1] = 1.0
    assert experiments._dataset_identity(original) != experiments._dataset_identity(changed)


def test_worker_reloads_changed_checkpoint_at_the_same_path(tmp_path, monkeypatch):
    path = tmp_path / "gate.pt"
    path.write_bytes(b"first checkpoint")
    loads = []

    def load(filename):
        model = object()
        loads.append((filename, model))
        return model

    monkeypatch.setattr(experiments, "_MODEL_CACHE", {})
    monkeypatch.setattr(experiments.ConsequenceEnsemble, "load", load)
    first = experiments._cached_model(path)
    assert experiments._cached_model(path) is first
    path.write_bytes(b"second checkpoint")
    second = experiments._cached_model(path)
    assert second is not first and len(loads) == 2
    assert experiments._cached_model(path) is second
    assert len(experiments._MODEL_CACHE) == 1
