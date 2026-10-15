"""Invalid protocols fail before expensive data generation/native initialization."""

from pathlib import Path

import pytest

from timelyhelp.config import load_config, validate_config


@pytest.mark.parametrize(
    "key,value",
    [
        ("dt", float("nan")),
        ("horizon", float("inf")),
        ("request_expiry", -1),
        ("max_queries", -1),
        ("camera_width", 1.5),
    ],
)
def test_invalid_observation_clock_and_budget_rejected(key, value):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["simulation"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_nonfinite_latency_and_empty_experimental_grid_rejected():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["generation"]["delays"] = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        validate_config(config)
    config["generation"]["delays"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_config(config)


def test_backend_override_revalidates_physics_clock_and_unknown_setting():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["simulation"]["dt"] = 0.015
    assert validate_config(config)["backend"] == "reference"
    config["backend"] = "isaac"
    with pytest.raises(ValueError, match="multiple"):
        validate_config(config)
    config["backend"] = "reference"
    config["training_seed"] = 13
    with pytest.raises(ValueError, match="Unknown"):
        validate_config(config)


@pytest.mark.parametrize(
    "settings",
    [
        {"oracle_enabled": "yes"},
        {"oracle_replications": 0},
        {"oracle_max_decisions": False},
        {"oracle_decision_interval_s": float("nan")},
        {"oracle_future_in_policy": True},
    ],
)
def test_invalid_privileged_diagnostic_protocol_rejected(settings):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["diagnostics"] = settings
    with pytest.raises(ValueError):
        validate_config(config)
