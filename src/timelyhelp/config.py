"""Validated experiment configuration and immutable run provenance."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .types import Candidate, SimConfig


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def validate_config(config):
    """Reject invalid protocols before generating data or starting native workers."""
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a mapping")
    unknown = set(config) - {
        "mode",
        "backend",
        "training_seeds",
        "scenario_seed",
        "simulation",
        "generation",
        "learning",
        "evaluation",
        "calibration",
        "ablations",
        "controllers",
        "diagnostics",
    }
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    if config.get("backend") not in {"reference", "isaac", "gazebo"}:
        raise ValueError("backend must be reference, isaac or gazebo")
    if config.get("mode") not in {"smoke", "pilot", "confirmatory"}:
        raise ValueError("mode must declare smoke, pilot or confirmatory")
    if len(set(config.get("training_seeds", []))) != len(config.get("training_seeds", [])):
        raise ValueError("Training seeds must be independent unique seeds")
    if config["mode"] == "confirmatory" and len(config.get("training_seeds", [])) < 5:
        raise ValueError("Confirmatory protocol requires at least five training seeds")
    if not config.get("training_seeds"):
        raise ValueError("At least one training seed is required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in config["training_seeds"]
    ):
        raise ValueError("Training seeds must be integers")
    simulation = SimConfig.from_dict(config.get("simulation", {}))
    positive = (
        "dt",
        "horizon",
        "commitment",
        "robot_radius",
        "max_speed",
        "acceleration",
        "braking",
        "sensor_range",
        "camera_width",
        "camera_height",
        "track_timeout",
        "max_reply_age",
        "request_expiry",
        "history_length",
    )
    for key in positive:
        value = getattr(simulation, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"simulation.{key} must be finite and positive")
    nonnegative = (
        "cautious_speed",
        "fusion_time",
        "max_queries",
        "near_miss_distance",
        "collision_weight",
        "failure_weight",
        "time_weight",
        "query_cost",
        "detector_noise",
    )
    for key in nonnegative:
        value = getattr(simulation, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"simulation.{key} must be finite and nonnegative")
    if not 0 < simulation.camera_fov < math.pi or simulation.cautious_speed > simulation.max_speed:
        raise ValueError("Invalid field of view or cautious speed")
    for key in ("camera_width", "camera_height", "history_length", "max_queries"):
        if isinstance(getattr(simulation, key), bool) or not isinstance(
            getattr(simulation, key), int
        ):
            raise ValueError(f"simulation.{key} must be an integer")
    if config["backend"] != "reference" and not math.isclose(
        simulation.dt / 0.01, round(simulation.dt / 0.01)
    ):
        raise ValueError("Native physics requires dt to be an integer multiple of 0.01 seconds")
    if set(simulation.source_latency) != {"infrastructure", "supervisor"}:
        raise ValueError("Advertised latency priors must cover both sources")
    for values in simulation.source_latency.values():
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Advertised latency samples must be finite and nonnegative")
    generation = config.get("generation", {})
    for split in ("train", "development", "calibration", "test"):
        count = generation.get("episodes", {}).get(split, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Missing independent {split} episodes")
    for key in ("snapshots_per_episode", "replications"):
        value = generation.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"generation.{key} must be a positive integer")
    times = generation.get("snapshot_times", [0.0])
    if len(times) < generation.get("snapshots_per_episode", 1) or any(
        not math.isfinite(float(t)) or not 0 <= t < simulation.horizon for t in times
    ):
        raise ValueError("Every snapshot requires a finite time before the mission horizon")
    delays = generation.get("delays", [])
    if not delays or any(not math.isfinite(float(x)) or float(x) < 0 for x in delays):
        raise ValueError("Delay probes must be nonempty, finite and nonnegative")
    for source in generation.get("sources", []):
        if source == "none":
            raise ValueError(
                "No-query targets derive from W branches, not extra source simulations"
            )
        for bridge in generation.get("bridges", []):
            Candidate(source, bridge)
    if not generation.get("sources") or not generation.get("bridges"):
        raise ValueError("Generation needs at least one source and bridge")
    controllers = config.get("controllers", ["local"])
    if (
        not controllers
        or len(set(controllers)) != len(controllers)
        or set(controllers) - {"local", "reactive"}
    ):
        raise ValueError("Controllers must be a nonempty unique subset of local/reactive")
    for key in ("epochs", "ensemble_size", "hidden_dim", "batch_size"):
        value = config.get("learning", {}).get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"learning.{key} must be a positive integer")
    for key in ("episodes", "transfer_episodes", "bootstrap_samples"):
        value = config.get("evaluation", {}).get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    calibration = config.get("calibration", {})
    if not calibration.get("risk_limits") or not calibration.get("query_penalties"):
        raise ValueError("Independent calibration needs risk and cost threshold grids")
    if any(not math.isfinite(float(x)) or not 0 < x <= 1 for x in calibration["risk_limits"]):
        raise ValueError("Calibration risk limits must lie in (0, 1]")
    if any(not math.isfinite(float(x)) or x < 0 for x in calibration["query_penalties"]):
        raise ValueError("Calibration query penalties must be finite and nonnegative")
    if not 0 < calibration.get("max_collision_rate", 0.05) < 1:
        raise ValueError("Calibration maximum collision rate must lie in (0, 1)")
    diagnostics = config.get("diagnostics", {})
    if not isinstance(diagnostics, dict) or set(diagnostics) - {
        "oracle_enabled",
        "oracle_episodes",
        "oracle_max_decisions",
        "oracle_replications",
        "oracle_decision_interval_s",
    }:
        raise ValueError("Unknown privileged diagnostic settings")
    if not isinstance(diagnostics.get("oracle_enabled", False), bool):
        raise ValueError("diagnostics.oracle_enabled must be boolean")
    for key, default in (
        ("oracle_max_decisions", 1),
        ("oracle_replications", 1),
        ("oracle_episodes", 6),
    ):
        value = diagnostics.get(key, default)
        if key == "oracle_max_decisions" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"diagnostics.{key} must be a positive integer")
    interval = diagnostics.get("oracle_decision_interval_s")
    if interval is not None and (not math.isfinite(interval) or interval <= 0):
        raise ValueError("Oracle decision interval must be finite and positive")
    return config


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def provenance(config, *, stage):
    packages = {}
    for name in ("timelyhelp", "numpy", "torch", "scipy", "PyYAML", "matplotlib"):
        packages[name] = version(name)
    root = Path(__file__).resolve().parents[2]
    result = {
        "stage": stage,
        "config": config,
        "config_sha256": digest(config),
        "timestamp": datetime.now(ZoneInfo("America/Winnipeg")).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "methodology": "TimelyHelp_Research_Brief.md",
        "results_are_smoke": config["mode"] == "smoke",
    }
    brief = root / "TimelyHelp_Research_Brief.md"
    result["brief_sha256"] = hashlib.sha256(brief.read_bytes()).hexdigest()
    for key, cwd in (("git_commit", root), ("dynanav_commit", root / "third_party/TIC-VLA")):
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
        )
        result[key] = process.stdout.strip() if process.returncode == 0 else "uncommitted"
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src").rglob("*.py"))
    }
    result["source_sha256"] = digest(source_hashes)
    result["source_files"] = source_hashes
    return result
