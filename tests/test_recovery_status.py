"""Truthful recovery status without launching native simulation workers."""

import json
import os
import queue
from types import SimpleNamespace

import pytest

from timelyhelp import experiments, sim_pool
from timelyhelp.config import write_json
from timelyhelp.sim_pool import IsolatedIsaacPool, _write_worker_status, execute_jobs


def _read(path):
    return json.loads(path.read_text())


def _mock_stages(monkeypatch, output, *, fail_at=None, error=None):
    def stage_directory(path, config, name):
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(experiments, "_stage", stage_directory)
    observed = []
    for name in ("generate", "train", "evaluate", "report"):

        def stage(*args, _name=name, **kwargs):
            current = _read(output / "run_status.json")
            assert current["stage"] == _name
            assert current["status"] == "running"
            assert current["resumable_stage"] == _name
            assert current["pid"] == os.getpid()
            observed.append(_name)
            if fail_at == _name:
                raise error
            return {"stage_artifact": _name}

        function = "generate_data" if name == "generate" else f"{name}_pipeline"
        monkeypatch.setattr(experiments, function, stage)
    return observed


def test_pipeline_records_every_stage_and_terminal_completion(tmp_path, monkeypatch):
    seen = _mock_stages(monkeypatch, tmp_path)
    manifest = experiments.run_pipeline({"backend": "reference", "mode": "pilot"}, tmp_path)
    status = _read(tmp_path / "run_status.json")
    assert seen == ["generate", "train", "evaluate", "report"]
    assert [item["stage"] for item in status["history"]] == [
        "starting",
        *seen,
        "complete",
    ]
    assert status["stage"] == status["status"] == "complete"
    assert status["resumable_stage"] is None
    assert status["started_unix_time"] <= status["finished_unix_time"]
    assert status["updated_at_utc"].endswith("Z")
    assert status["backend"] == "reference" and status["parallel_gpus"] == []
    assert manifest == _read(tmp_path / "pipeline_manifest.json")


@pytest.mark.parametrize("stage", ["generate", "train", "evaluate", "report"])
def test_pipeline_failure_retains_actual_resume_stage(tmp_path, monkeypatch, stage):
    _mock_stages(monkeypatch, tmp_path, fail_at=stage, error=RuntimeError("pair mismatch"))
    with pytest.raises(RuntimeError, match="pair mismatch"):
        experiments.run_pipeline({"backend": "reference", "mode": "pilot"}, tmp_path)
    status = _read(tmp_path / "run_status.json")
    assert status["stage"] == status["status"] == "failed"
    assert status["resumable_stage"] == stage
    assert status["history"][-2]["stage"] == stage
    assert status["error"]["type"] == "RuntimeError"
    assert status["error"]["message"] == "pair mismatch"
    assert "RuntimeError: pair mismatch" in status["error"]["traceback"]
    assert not (tmp_path / "pipeline_manifest.json").exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(143)])
def test_pipeline_records_interruption_without_swallowing_it(tmp_path, monkeypatch, error):
    _mock_stages(monkeypatch, tmp_path, fail_at="generate", error=error)
    with pytest.raises(type(error)):
        experiments.run_pipeline({"backend": "reference", "mode": "pilot"}, tmp_path)
    status = _read(tmp_path / "run_status.json")
    assert status["stage"] == status["status"] == "interrupted"
    assert status["resumable_stage"] == "generate"


def _fake_pool(tmp_path):
    """Synchronous queue events exercise the real scheduler and shard validation."""
    pool = object.__new__(IsolatedIsaacPool)
    pool.closed = False
    pool.status_dir = tmp_path
    pool.gpus = (0,)
    pool.events = queue.Queue()
    pool.processes = {0: SimpleNamespace(pid=123, is_alive=lambda: True)}
    pool.restarts = {0: 0}
    dispatches = []
    state = {"fail": True}

    def submit(job):
        dispatches.append(job["id"])
        pool.events.put({"event": "started", "id": job["id"], "gpu": 0})
        if state["fail"]:
            event = {"event": "failed", "error": "prearrival pair mismatch"}
        else:
            write_json(
                job["shard"],
                {"complete": True, "work_sha256": job["work_sha256"], "gpu": 0},
            )
            event = {"event": "finished"}
        pool.events.put({**event, "id": job["id"], "gpu": 0})

    pool.queues = {0: SimpleNamespace(put=submit)}
    return pool, dispatches, state


def test_pairing_failure_is_not_retried_and_resume_clears_stale_status(tmp_path):
    pool, dispatches, state = _fake_pool(tmp_path)
    jobs = [
        {
            "id": f"episode_{index}",
            "kind": "generate_episode",
            "shard": str(tmp_path / f"shard_{index}.json"),
            "work_sha256": f"hash_{index}",
        }
        for index in range(3)
    ]
    write_json(jobs[0]["shard"], {"complete": True, "work_sha256": "hash_0"})
    with pytest.raises(RuntimeError, match="prearrival pair mismatch"):
        pool.run(jobs)
    failed = _read(tmp_path / "progress.json")
    assert failed["status"] == "failed"
    assert failed["completed"] == 1 and failed["total"] == 3
    assert failed["failed_job_ids"] == ["episode_1"]
    assert failed["inflight"] == {}
    assert failed["failures"] == 1
    assert dispatches == ["episode_1"]
    state["fail"] = False
    assert len(pool.run(jobs)) == 3
    complete = _read(tmp_path / "progress.json")
    assert complete["status"] == "complete"
    assert complete["completed"] == complete["total"] == 3
    assert complete["failed_job_ids"] == [] and complete["failures"] == 0
    assert complete["errors"] == [] and complete["inflight"] == {}
    assert dispatches == ["episode_1", "episode_1", "episode_2"]
    # A completely cached resume must also replace stale running/failed status.
    assert len(pool.run(jobs)) == 3
    assert _read(tmp_path / "progress.json")["status"] == "complete"
    assert dispatches == ["episode_1", "episode_1", "episode_2"]


def test_worker_completion_failure_and_restart_clear_active_job(tmp_path):
    _write_worker_status(tmp_path, 0, "running", job="episode_159")
    assert _read(tmp_path / "gpu_0.json")["job"] == "episode_159"
    _write_worker_status(tmp_path, 0, "failed", last_job="episode_159", error="pair mismatch")
    failed = _read(tmp_path / "gpu_0.json")
    assert failed["job"] is None and failed["last_job"] == "episode_159"
    assert failed["error"] == "pair mismatch"
    _write_worker_status(tmp_path, 0, "running", job="episode_159")
    assert "error" not in _read(tmp_path / "gpu_0.json")
    _write_worker_status(tmp_path, 0, "complete", last_job="episode_159")
    complete = _read(tmp_path / "gpu_0.json")
    assert complete["job"] is None and complete["last_job"] == "episode_159"
    assert "error" not in complete


def test_exhausted_worker_exit_retains_failed_job_without_claiming_activity(tmp_path):
    pool, _, _ = _fake_pool(tmp_path)
    pool._failed_job_ids = set()
    pool.restarts[0] = 2
    pool.processes[0] = SimpleNamespace(pid=123, exitcode=-9, is_alive=lambda: False)
    job = {"id": "episode_159"}
    inflight = {job["id"]: job}
    gpu_job = {0: job}
    failures = []
    pool._reap(inflight, gpu_job, failures)
    assert inflight == gpu_job == {}
    assert len(failures) == 1
    assert pool._failed_job_ids == {job["id"]}
    status = _read(tmp_path / "gpu_0.json")
    assert status["status"] == "failed" and status["job"] is None
    assert status["last_job"] == job["id"] and status["pid"] == 123


def test_inline_job_failure_does_not_leave_running_worker_status(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMELYHELP_GPU", raising=False)
    job = {
        "id": "episode_159",
        "status_dir": str(tmp_path),
        "shard": str(tmp_path / "missing.json"),
        "work_sha256": "expected",
    }

    def fail(job):
        raise ValueError("pair mismatch")

    monkeypatch.setattr(experiments, "dispatch_simulation_job", fail)
    with pytest.raises(ValueError, match="pair mismatch"):
        execute_jobs({"backend": "reference"}, [job])
    status = _read(tmp_path / "gpu_inline.json")
    assert status["status"] == "failed" and status["job"] is None
    assert status["last_job"] == job["id"]
    assert "ValueError: pair mismatch" in status["error"]


def test_tick_logging_preserves_branch_progress_and_completion_clears_it(tmp_path, monkeypatch):
    from timelyhelp.backends.native import NativeSimulator

    monkeypatch.setenv("TIMELYHELP_GPU", "0")
    monkeypatch.setenv("TIMELYHELP_STATUS_DIR", str(tmp_path))
    monkeypatch.setattr(sim_pool, "current_job_id", "episode_159")
    monkeypatch.setattr(sim_pool, "current_job_progress", None)
    monkeypatch.setattr(NativeSimulator, "_physics_step", lambda self, control: None)
    monkeypatch.setattr(NativeSimulator, "_timelyhelp_tick_log", False, raising=False)
    sim_pool._install_native_tick_log()
    message = "snapshot t=1.50s branch 17/60 infrastructure/hold/delay=0.75/rep=1"
    sim_pool.touch_status({"id": "episode_159", "status_dir": str(tmp_path)}, message)
    assert _read(tmp_path / "gpu_0.json")["job_progress"] == message
    fake = SimpleNamespace(_tick_logs=49, tick=50, time=5.0, pose=[1.0, 2.0, 0.0])
    NativeSimulator._physics_step(fake, None)
    status = _read(tmp_path / "gpu_0.json")
    assert status["status"] == "running" and status["job"] == "episode_159"
    assert status["message"].startswith("physics tick=50")
    assert status["job_progress"] == message
    sim_pool._write_worker_status(tmp_path, 0, "complete", last_job="episode_159")
    assert _read(tmp_path / "gpu_0.json")["job_progress"] is None
