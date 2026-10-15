"""Real process locks protect pilot artifacts and release on every exit path."""

import json
import os
import subprocess
import sys

import pytest

from timelyhelp import experiments


def test_concurrent_pipeline_is_rejected_before_mutating_run_artifacts(tmp_path, monkeypatch):
    # A separate Python process holds the same kernel lock a pipeline would hold.
    # Its stdin is an explicit lifetime barrier, so this test needs no sleeps.
    code = """
import fcntl, json, os, sys
with open(sys.argv[1], 'a+') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock.write(json.dumps({'pid': os.getpid()}))
    lock.flush()
    print('locked', flush=True)
    sys.stdin.readline()
"""
    lock_path = tmp_path / ".pipeline.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        provenance = tmp_path / "provenance.json"
        status = tmp_path / "run_status.json"
        provenance.write_bytes(b"original provenance")
        status.write_bytes(b"original status")
        original_owner = lock_path.read_bytes()

        def forbidden_stage(*args, **kwargs):
            pytest.fail("A rejected concurrent pipeline reached provenance mutation")

        monkeypatch.setattr(experiments, "_stage", forbidden_stage)
        with pytest.raises(RuntimeError, match=f"already locked.*{holder.pid}"):
            experiments.run_pipeline({"backend": "reference", "mode": "pilot"}, tmp_path)
        assert provenance.read_bytes() == b"original provenance"
        assert status.read_bytes() == b"original status"
        assert lock_path.read_bytes() == original_owner
    finally:
        holder.communicate("\n", timeout=5)
    assert holder.returncode == 0
    # Stale owner metadata remains harmless once its actual lock is released.
    with experiments._pipeline_output_lock(tmp_path):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()


@pytest.mark.parametrize("error", [None, RuntimeError("failed"), KeyboardInterrupt()])
def test_pipeline_releases_output_lock_after_success_failure_or_interrupt(
    tmp_path, monkeypatch, error
):
    def body(config, output):
        # A second descriptor in the same process also cannot acquire this lock.
        with pytest.raises(RuntimeError, match="already locked"):
            with experiments._pipeline_output_lock(output):
                pytest.fail("Concurrent ownership was granted")
        if error is not None:
            raise error
        return {"completed": True}

    monkeypatch.setattr(experiments, "_unlocked_run_pipeline", body)
    if error is None:
        assert experiments.run_pipeline({}, tmp_path) == {"completed": True}
    else:
        with pytest.raises(type(error)):
            experiments.run_pipeline({}, tmp_path)
    lock_path = tmp_path / ".pipeline.lock"
    assert lock_path.is_file()
    with experiments._pipeline_output_lock(tmp_path):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
