"""The read-only monitor distinguishes stale records from live work."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def monitor():
    path = Path(__file__).resolve().parents[1] / "scripts/pilot_status.py"
    spec = importlib.util.spec_from_file_location("pilot_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_dead_pid_marks_recorded_running_as_stopped(tmp_path, monkeypatch, monitor):
    monkeypatch.setattr(monitor, "pid_is_alive", lambda pid: False)
    _write(
        tmp_path / "run_status.json",
        {"pid": 123, "stage": "generate", "status": "running", "unix_time": 100},
    )
    _write(
        tmp_path / "workers/progress.json",
        {
            "completed": 159,
            "total": 200,
            "failures": 4,
            "job_kinds": ["generate_episode"],
            "status": "running",
            "unix_time": 100,
        },
    )
    _write(
        tmp_path / "workers/gpu_0.json",
        {
            "gpu": 0,
            "pid": 456,
            "status": "running",
            "job": "generate/reactive/train/0059",
            "unix_time": 100,
        },
    )
    text = monitor.format_status(tmp_path, now=160)
    assert text.count("STOPPED/stale") == 3
    assert "159/200 completed" in text and "generation episodes" in text
    assert "1m00s ago" in text and "COMPLETE" not in text


def test_live_gpu_shows_branch_and_physics_progress(tmp_path, monkeypatch, monitor):
    monkeypatch.setattr(monitor, "pid_is_alive", lambda pid: pid in {123, 456})
    _write(
        tmp_path / "run_status.json",
        {"pid": 123, "stage": "train", "status": "running", "unix_time": 100},
    )
    _write(
        tmp_path / "workers/progress.json",
        {
            "pid": 123,
            "status": "running",
            "completed": 5,
            "total": 15,
            "failures": 0,
            "job_kinds": ["calibration_episode"],
            "unix_time": 100,
        },
    )
    _write(
        tmp_path / "workers/gpu_0.json",
        {
            "gpu": 0,
            "pid": 456,
            "status": "running",
            "job": "calibration/local/11/timelyhelp",
            "job_progress": "branch 17/60",
            "message": "physics tick=50 t=5.00s",
            "unix_time": 105,
        },
    )
    _write(
        tmp_path / "models/progress.json",
        {"completed_controller": "local", "training_seed": 11, "variant": "timelyhelp"},
    )
    text = monitor.format_status(tmp_path, now=110)
    assert "RUNNING (PID 123 alive)" in text
    assert "calibration missions" in text and "5/15 completed" in text
    assert "Branch progress: branch 17/60" in text
    assert "Latest update: physics tick=50" in text
    assert "heartbeat 5s ago" in text
    assert "Last completed model/calibration" in text
    assert "STOPPED" not in text


def test_dead_process_does_not_override_explicit_completion(tmp_path, monkeypatch, monitor):
    monkeypatch.setattr(monitor, "pid_is_alive", lambda pid: False)
    _write(tmp_path / "run_status.json", {"pid": 123, "stage": "complete", "status": "complete"})
    assert "Pipeline: COMPLETE (PID 123 dead)" in monitor.format_status(tmp_path)


def test_zombie_is_not_a_live_worker(monkeypatch, monitor):
    monkeypatch.setattr(monitor.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(monitor.Path, "read_text", lambda path: "123 (worker with spaces) Z 1 2 3")
    assert not monitor.pid_is_alive(123)
