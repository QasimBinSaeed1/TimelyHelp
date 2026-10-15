"""Scheduler checks: one pilot, stable seeds, and shard resume. No Isaac process."""

import json
from pathlib import Path
import pytest

from timelyhelp.config import load_config
from timelyhelp.experiments import generate_data, plan_generation_jobs
from timelyhelp import experiments
from timelyhelp.sim_pool import parallel_simulations, read_valid_shard


def test_pilot_episodes_are_partitioned_once():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/pilot.yaml")
    jobs = plan_generation_jobs(config)
    assert len(jobs) == 2 * (60 + 15 + 15 + 10)
    assert len({job["id"] for job in jobs}) == len(jobs)
    local = [job for job in jobs if job["controller"] == "local"]
    train = [job for job in local if job["split"] == "train"]
    assert [job["seed"] for job in train[:3]] == [20261010, 20261011, 20261012]
    development = next(job for job in local if job["split"] == "development")
    assert development["seed"] == 20261010 + 10000
    assert len({(job["controller"], job["seed"]) for job in jobs}) == len(jobs)
    reactive = [job for job in jobs if job["controller"] == "reactive" and job["split"] == "train"]
    assert [job["seed"] for job in reactive[:3]] == [job["seed"] for job in train[:3]]


def test_parallel_scheduler_is_isaac_only(monkeypatch):
    monkeypatch.delenv("TIMELYHELP_SERIAL", raising=False)
    monkeypatch.setenv("TIMELYHELP_GPUS", "0,1,2,3")
    assert parallel_simulations({"backend": "isaac"})
    assert not parallel_simulations({"backend": "reference"})
    assert not parallel_simulations({"backend": "gazebo"})
    monkeypatch.setenv("TIMELYHELP_SERIAL", "1")
    assert not parallel_simulations({"backend": "isaac"})


def test_completed_generation_shards_resume_without_new_episodes(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["simulation"].update(horizon=0.4, max_queries=1, commitment=0.2)
    config["generation"].update(
        episodes={split: 1 for split in ("train", "development", "calibration", "test")},
        snapshot_times=[0.1],
        delays=[0.0],
        bridges=["cautious"],
        sources=["infrastructure"],
        replications=1,
    )
    config["controllers"] = ["local"]
    calls = {"n": 0}
    original = experiments._generate_episode_job

    def counted(job):
        calls["n"] += 1
        return original(job)

    monkeypatch.setattr(experiments, "_generate_episode_job", counted)
    first = generate_data(config, tmp_path / "data")
    assert first["raw_rows"] == 4
    assert calls["n"] == 4
    assert json.loads((tmp_path / "data/restoration_audits.json").read_text())[0]["passed"]
    again = generate_data(config, tmp_path / "data")
    assert again["raw_rows"] == 4
    assert calls["n"] == 4
    (tmp_path / "data/manifest.json").unlink()
    resumed = generate_data(config, tmp_path / "data")
    assert resumed["raw_rows"] == 4
    assert calls["n"] == 4
    shard = next((tmp_path / "data/shards").rglob("*.json"))
    saved = json.loads(shard.read_text())
    assert saved["complete"]
    assert "gpu" not in saved or saved.get("work_sha256")
    rejected = dict(saved)
    rejected["work_sha256"] = "different"
    shard.write_text(json.dumps(rejected))
    assert read_valid_shard({"shard": str(shard), "work_sha256": saved["work_sha256"]}) is None


def test_episode_checksum_depends_on_seed_not_output_path(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    job = plan_generation_jobs(config)[0]
    left = experiments._prepare_generation_job(config, tmp_path / "a", job, False)
    right = experiments._prepare_generation_job(config, tmp_path / "b", job, False)
    assert left["work_sha256"] == right["work_sha256"]
    assert left["shard"] != right["shard"]
    audited = experiments._prepare_generation_job(config, tmp_path / "a", job, True)
    assert audited["work_sha256"] != left["work_sha256"]


@pytest.mark.parametrize(
    "changed",
    [
        "src/timelyhelp/experiments.py",
        "src/timelyhelp/types.py",
        "src/timelyhelp/config.py",
        "src/timelyhelp/sim_pool.py",
        "src/timelyhelp/learning/data.py",
    ],
)
def test_scientific_cache_identity_covers_orchestration_and_targets(tmp_path, monkeypatch, changed):
    from timelyhelp import sim_pool

    monkeypatch.setattr(sim_pool, "ROOT", tmp_path)
    monkeypatch.setattr(sim_pool, "_CODE_HASHES", {})
    for relative in set(sim_pool._MISSION_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    before = sim_pool.code_hash("mission")
    (tmp_path / changed).write_text("changed scientific computation")
    sim_pool._CODE_HASHES.clear()
    assert sim_pool.code_hash("mission") != before
