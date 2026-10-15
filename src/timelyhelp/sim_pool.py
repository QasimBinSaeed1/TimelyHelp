"""Isolated Isaac workers for one shared, resumable simulation schedule.

Each worker is a separate process with ``TIMELYHELP_GPU`` set to exactly one of
the configured devices. Jobs are episodes, calibration rollouts, or matched
missions. A job runs on one GPU. The pool never replicates a pilot across cards.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import time
import traceback

from .config import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
GENERATION_LOGIC = 1
CALIBRATION_LOGIC = 1
MISSION_LOGIC = 1
ORACLE_LOGIC = 1

_GENERATION_FILES = (
    "src/timelyhelp/config.py",
    "src/timelyhelp/types.py",
    "src/timelyhelp/experiments.py",
    "src/timelyhelp/sim_pool.py",
    "src/timelyhelp/simulation/reference.py",
    "src/timelyhelp/simulation/scenarios.py",
    "src/timelyhelp/backends/native.py",
    "src/timelyhelp/backends/isaac_worker.py",
    "src/timelyhelp/backends/gazebo_worker.py",
    "src/timelyhelp/backends/worker_common.py",
    "src/timelyhelp/backends/worlds.py",
    "src/timelyhelp/counterfactual.py",
    "src/timelyhelp/perception.py",
    "src/timelyhelp/control.py",
    "src/timelyhelp/sources.py",
    "src/timelyhelp/audits.py",
)
_MISSION_FILES = _GENERATION_FILES + (
    "src/timelyhelp/evaluation.py",
    "src/timelyhelp/baselines.py",
    "src/timelyhelp/oracle.py",
    "src/timelyhelp/learning/scheduler.py",
    "src/timelyhelp/learning/model.py",
    "src/timelyhelp/learning/data.py",
)

_POOL = None
current_job_id = None
current_job_progress = None
_CODE_HASHES: dict[str, str] = {}


def configured_gpus():
    """Physical GPU indices. ``TIMELYHELP_GPUS`` overrides the four-card default."""
    raw = os.environ.get("TIMELYHELP_GPUS", "0,1,2,3")
    gpus = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid TIMELYHELP_GPUS entry {part!r}")
        gpus.append(int(part))
    if not gpus:
        raise ValueError("TIMELYHELP_GPUS must list at least one GPU")
    if len(set(gpus)) != len(gpus):
        raise ValueError("TIMELYHELP_GPUS contains duplicate indices")
    return tuple(gpus)


def parallel_simulations(config) -> bool:
    """Isaac episodes are partitioned across GPUs. Reference and Gazebo stay in-process."""
    if config.get("backend") != "isaac":
        return False
    if os.environ.get("TIMELYHELP_SERIAL", "") == "1":
        return False
    return len(configured_gpus()) >= 2


def code_hash(scope: str) -> str:
    cached = _CODE_HASHES.get(scope)
    if cached is not None:
        return cached
    if scope == "generation":
        files = _GENERATION_FILES
    elif scope == "mission":
        files = _MISSION_FILES
    else:
        raise ValueError(f"Unknown code hash scope {scope}")
    digest_bytes = hashlib.sha256()
    for relative in files:
        path = ROOT / relative
        digest_bytes.update(relative.encode())
        digest_bytes.update(path.read_bytes())
    value = digest_bytes.hexdigest()
    _CODE_HASHES[scope] = value
    return value


def status_directory(stage_output) -> Path:
    stage_output = Path(stage_output)
    if stage_output.name in {"data", "models", "evaluation", "report"}:
        return stage_output.parent / "workers"
    return stage_output / "workers"


# Shards written before the per-camera capture copy/isolation fix stay valid.
# Those changes stop a false W/Q RGB-row mismatch; they do not rewrite episodes
# that already matched.
ACCEPTED_GENERATION_CODES = ("763419a1d629d03003ee0e8199cb5e6905177c669e7214a5f07af871bcece1fc",)


def work_hash(scope: str, config, identity: dict, code: str | None = None) -> str:
    logic = {
        "generation": GENERATION_LOGIC,
        "calibration": CALIBRATION_LOGIC,
        "mission": MISSION_LOGIC,
        "oracle": ORACLE_LOGIC,
    }[scope]
    code_scope = "generation" if scope in {"generation", "calibration"} else "mission"
    return digest(
        {
            "logic": logic,
            "code": code if code is not None else code_hash(code_scope),
            "config": digest(config),
            "identity": identity,
        }
    )


def read_valid_shard(job):
    path = Path(job["shard"])
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    accepted = {job["work_sha256"], *job.get("legacy_work_hashes", ())}
    if data.get("work_sha256") not in accepted or not data.get("complete"):
        return None
    return data


def _write_shard(job, payload, gpu):
    saved = {
        "complete": True,
        "work_sha256": job["work_sha256"],
        "source_code_sha256": code_hash(
            "generation"
            if job["kind"] in {"generate_episode", "calibration_episode"}
            else "mission"
        ),
        "gpu": None if gpu is None else int(gpu) if str(gpu).isdigit() else gpu,
        "job_id": job["id"],
        **payload,
    }
    write_json(job["shard"], saved)
    return saved


def _write_worker_status(status_dir, gpu, status, *, job=None, last_job=None, **details):
    """Replace worker state atomically, so idle/failed workers have no active job."""
    write_json(
        Path(status_dir) / f"gpu_{gpu}.json",
        {
            "gpu": gpu,
            "pid": os.getpid(),
            "status": status,
            "job": job,
            "job_progress": current_job_progress if status == "running" else None,
            "last_job": last_job,
            "unix_time": time.time(),
            **details,
        },
    )


def touch_status(job, message):
    """Record the active job without claiming more than one GPU."""
    global current_job_progress
    current_job_progress = message
    gpu = os.environ.get("TIMELYHELP_GPU", "inline")
    status_dir = Path(job["status_dir"])
    status_dir.mkdir(parents=True, exist_ok=True)
    _write_worker_status(status_dir, gpu, "running", job=job["id"], message=message)
    print(f"[gpu {gpu}] {job['id']}: {message}", flush=True)


def _install_native_tick_log():
    from .backends.native import NativeSimulator

    if getattr(NativeSimulator, "_timelyhelp_tick_log", False):
        return
    original = NativeSimulator._physics_step

    def _physics_step(self, control):
        original(self, control)
        count = getattr(self, "_tick_logs", 0) + 1
        self._tick_logs = count
        if count % 50 != 0:
            return
        gpu = os.environ.get("TIMELYHELP_GPU", "?")
        message = (
            f"physics tick={self.tick} t={self.time:.2f}s job={current_job_id} "
            f"pose=({self.pose[0]:.2f},{self.pose[1]:.2f})"
        )
        print(f"[gpu {gpu}] {message}", flush=True)
        status_dir = os.environ.get("TIMELYHELP_STATUS_DIR")
        if status_dir:
            _write_worker_status(
                status_dir,
                gpu,
                "running" if current_job_id is not None else "idle",
                job=current_job_id,
                message=message,
                simulation_time=float(self.time),
            )

    NativeSimulator._physics_step = _physics_step
    NativeSimulator._timelyhelp_tick_log = True


def _worker_entry(gpu, jobs, events, status_dir):
    global current_job_id, current_job_progress
    os.setsid()
    os.environ["TIMELYHELP_GPU"] = str(gpu)
    # Isaac selects the physical device index. A parent CUDA mask would renumber it.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["TIMELYHELP_STATUS_DIR"] = status_dir
    native_root = Path(status_dir) / f"gpu_{gpu}" / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    os.environ["TIMELYHELP_NATIVE_CACHE"] = str(native_root)
    log_path = Path(status_dir) / f"gpu_{gpu}.log"
    log_file = log_path.open("a", buffering=1)

    class _Tee:
        def write(self, data):
            log_file.write(data)
            log_file.flush()
            if not isinstance(data, (bytes, bytearray)):
                data = str(data).encode(errors="replace")
            try:
                os.write(1, data)
            except OSError:
                pass

        def flush(self):
            log_file.flush()

    import sys

    sys.stdout = _Tee()
    sys.stderr = _Tee()
    print(f"worker ready gpu={gpu} pid={os.getpid()} log={log_path}", flush=True)
    _write_worker_status(status_dir, gpu, "idle", message="worker ready")
    _install_native_tick_log()
    while True:
        job = jobs.get()
        if job is None:
            break
        current_job_id = job["id"]
        current_job_progress = None
        _write_worker_status(status_dir, gpu, "running", job=job["id"])
        events.put({"event": "started", "id": job["id"], "gpu": gpu, "pid": os.getpid()})
        try:
            cached = read_valid_shard(job)
            if cached is None:
                from .experiments import dispatch_simulation_job

                payload = dispatch_simulation_job(job)
                _write_shard(job, payload, gpu)
            _write_worker_status(status_dir, gpu, "complete", last_job=job["id"])
            events.put({"event": "finished", "id": job["id"], "gpu": gpu})
        except Exception:
            error = traceback.format_exc()
            _write_worker_status(status_dir, gpu, "failed", last_job=job["id"], error=error)
            events.put(
                {
                    "event": "failed",
                    "id": job["id"],
                    "gpu": gpu,
                    "error": error,
                }
            )
        finally:
            current_job_id = None
            current_job_progress = None
    from .backends.native import _close_pool

    _close_pool()
    _write_worker_status(status_dir, gpu, "stopped", message="worker shutdown complete")
    print(f"worker exit gpu={gpu}", flush=True)


class IsolatedIsaacPool:
    """Persistent conda workers. Each one owns the Isaac processes for a single GPU."""

    def __init__(self, gpus, status_dir):
        import multiprocessing as mp

        self.gpus = tuple(gpus)
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context("spawn")
        self.events = self.ctx.Queue()
        self.queues = {gpu: self.ctx.Queue() for gpu in self.gpus}
        self.processes = {}
        self.restarts = {gpu: 0 for gpu in self.gpus}
        self.closed = False
        available = _available_gpus()
        missing = [gpu for gpu in self.gpus if gpu not in available]
        if missing:
            raise RuntimeError(
                f"Requested GPUs {missing} are not visible; nvidia-smi has {sorted(available)}"
            )
        for gpu in self.gpus:
            self._start(gpu)

    def _start(self, gpu):
        process = self.ctx.Process(
            target=_worker_entry,
            name=f"timelyhelp-gpu-{gpu}",
            args=(gpu, self.queues[gpu], self.events, str(self.status_dir)),
            daemon=False,
        )
        process.start()
        self.processes[gpu] = process
        print(f"started isolated worker gpu={gpu} pid={process.pid}", flush=True)

    def run(self, jobs):
        if self.closed:
            raise RuntimeError("Isaac worker pool is closed")
        self._progress_started_at = time.time()
        self._progress_job_kinds = sorted({job.get("kind", "unknown") for job in jobs})
        self._failed_job_ids = set()
        self._write_progress(len(jobs), {}, {}, {}, [])
        try:
            return self._run_jobs(jobs)
        except BaseException as error:
            status = (
                "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            )
            snapshot = dict(self._progress_snapshot)
            snapshot.update(
                status=status,
                error={"type": type(error).__name__, "message": str(error)},
                unix_time=time.time(),
                finished_unix_time=time.time(),
                failed_job_ids=sorted(self._failed_job_ids),
            )
            write_json(self.status_dir / "progress.json", snapshot)
            raise

    def _run_jobs(self, jobs):
        results = {}
        pending = []
        by_id = {}
        for job in jobs:
            by_id[job["id"]] = job
            cached = read_valid_shard(job)
            if cached is not None:
                results[job["id"]] = cached
            else:
                pending.append(job)
        if len({job["id"] for job in jobs}) != len(jobs):
            raise ValueError("Simulation schedule contains duplicate job ids")
        if not pending:
            self._write_progress(len(jobs), results, {}, {}, [], status="complete")
            return [results[job["id"]] for job in jobs]
        print(
            f"simulation jobs: {len(results)} resumed, {len(pending)} remaining, "
            f"gpus={list(self.gpus)}",
            flush=True,
        )
        next_index = 0
        inflight = {}
        gpu_job = {}
        failures = []
        last_report = time.time()

        def assign():
            nonlocal next_index
            free = [
                gpu for gpu in self.gpus if gpu not in gpu_job and self.processes[gpu].is_alive()
            ]
            while not failures and next_index < len(pending) and free:
                gpu = free.pop(0)
                job = pending[next_index]
                next_index += 1
                inflight[job["id"]] = job
                gpu_job[gpu] = job
                self.queues[gpu].put(job)

        assign()
        self._write_progress(len(jobs), results, inflight, gpu_job, failures)
        while inflight or (next_index < len(pending) and not failures):
            try:
                event = self.events.get(timeout=5)
            except queue.Empty:
                self._reap(inflight, gpu_job, failures)
                assign()
                now = time.time()
                if now - last_report >= 30:
                    self._write_progress(len(jobs), results, inflight, gpu_job, failures)
                    last_report = now
                continue
            job_id = event.get("id")
            if event["event"] == "started":
                self._write_progress(len(jobs), results, inflight, gpu_job, failures)
                continue
            inflight.pop(job_id, None)
            gpu = event["gpu"]
            if gpu_job.get(gpu) is not None and gpu_job[gpu]["id"] == job_id:
                gpu_job.pop(gpu, None)
            if event["event"] == "finished":
                shard = read_valid_shard(by_id[job_id])
                if shard is None:
                    self._failed_job_ids.add(job_id)
                    failures.append(f"{job_id} finished without a valid shard")
                else:
                    results[job_id] = shard
                    print(
                        f"completed {job_id} on gpu {shard.get('gpu')} "
                        f"({len(results)}/{len(jobs)})",
                        flush=True,
                    )
            else:
                error = event.get("error") or f"{job_id} failed"
                self._failed_job_ids.add(job_id)
                failures.append(f"{job_id}: {error}")
                print(f"job failed {job_id}\n{failures[-1]}", flush=True)
            self._reap(inflight, gpu_job, failures)
            assign()
            self._write_progress(len(jobs), results, inflight, gpu_job, failures)
        if failures:
            raise RuntimeError(
                "Simulation job failed; completed shards are retained for resume.\n"
                + "\n".join(failures[:4])
            )
        missing = [job["id"] for job in jobs if job["id"] not in results]
        if missing:
            raise RuntimeError(f"Simulation schedule lost jobs: {missing[:8]}")
        self._write_progress(len(jobs), results, inflight, gpu_job, failures, status="complete")
        return [results[job["id"]] for job in jobs]

    def _reap(self, inflight, gpu_job, failures):
        for gpu, process in list(self.processes.items()):
            if process.is_alive():
                continue
            self.restarts[gpu] += 1
            job = gpu_job.pop(gpu, None)
            code = process.exitcode
            _write_worker_status(
                self.status_dir,
                gpu,
                "failed",
                last_job=job["id"] if job is not None else None,
                pid=process.pid,
                error=f"worker exited with code {code}",
            )
            log_path = self.status_dir / f"gpu_{gpu}.log"
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(errors="replace")[-1500:]
            if self.restarts[gpu] > 2:
                failures.append(f"GPU {gpu} worker exited {code} repeatedly\n{tail}")
                if job is not None:
                    self._failed_job_ids.add(job["id"])
                    inflight.pop(job["id"], None)
                continue
            print(f"restarting isolated worker gpu={gpu} after exit {code}", flush=True)
            self._start(gpu)
            if job is None or job["id"] not in inflight:
                continue
            if read_valid_shard(job) is not None:
                self.events.put({"event": "finished", "id": job["id"], "gpu": gpu})
                continue
            print(f"requeue {job['id']} on gpu {gpu} after worker exit", flush=True)
            gpu_job[gpu] = job
            self.queues[gpu].put(job)

    def _write_progress(self, total, results, inflight, gpu_job, failures, *, status=None):
        status = status or ("failed" if failures else "running")
        now = time.time()
        snapshot = {
            "status": status,
            "completed": len(results),
            "total": total,
            "inflight": {str(gpu): job["id"] for gpu, job in gpu_job.items() if job},
            "failures": len(failures),
            "failed_job_ids": sorted(self._failed_job_ids),
            "errors": failures[-4:],
            "worker_pids": {str(gpu): process.pid for gpu, process in self.processes.items()},
            "worker_alive": {
                str(gpu): process.is_alive() for gpu, process in self.processes.items()
            },
            "pid": os.getpid(),
            "job_kinds": self._progress_job_kinds,
            "started_unix_time": self._progress_started_at,
            "unix_time": now,
        }
        if status in {"complete", "failed", "interrupted"}:
            snapshot["finished_unix_time"] = now
        self._progress_snapshot = snapshot
        write_json(self.status_dir / "progress.json", snapshot)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for gpu in self.gpus:
            try:
                self.queues[gpu].put(None)
            except (ValueError, OSError, BrokenPipeError):
                pass
        deadline = time.time() + 20
        for process in self.processes.values():
            remaining = max(0.1, deadline - time.time())
            process.join(timeout=remaining)
            if process.is_alive() and process.pid:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
        deadline = time.time() + 10
        for process in self.processes.values():
            process.join(timeout=max(0.1, deadline - time.time()))
            if process.is_alive() and process.pid:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.join(timeout=5)


def _assert_spawnable_main():
    import sys

    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if not path or path.startswith("<") or not Path(path).is_file():
        raise RuntimeError(
            "Isolated GPU workers are spawned from the main program file. "
            "Launch with `python -m timelyhelp.cli`, not an interactive stdin or python -c script."
        )


def get_pool(status_dir):
    global _POOL
    _assert_spawnable_main()
    gpus = configured_gpus()
    if _POOL is None or _POOL.closed:
        _POOL = IsolatedIsaacPool(gpus, status_dir)
        atexit.register(close_pool)
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def execute_jobs(config, jobs):
    """Run jobs in order, reusing valid shards. Isaac uses one worker per GPU."""
    if not jobs:
        return []
    status_dir = Path(jobs[0]["status_dir"])
    if not parallel_simulations(config):
        results = []
        for index, job in enumerate(jobs):
            cached = read_valid_shard(job)
            if cached is not None:
                results.append(cached)
                continue
            from .experiments import dispatch_simulation_job

            touch_status(job, "running in-process")
            gpu = os.environ.get("TIMELYHELP_GPU", "inline")
            try:
                payload = dispatch_simulation_job(job)
                results.append(_write_shard(job, payload, os.environ.get("TIMELYHELP_GPU")))
            except BaseException as error:
                status = (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "failed"
                )
                _write_worker_status(
                    status_dir, gpu, status, last_job=job["id"], error=traceback.format_exc()
                )
                raise
            _write_worker_status(status_dir, gpu, "complete", last_job=job["id"])
            print(f"completed {job['id']} ({index + 1}/{len(jobs)})", flush=True)
        return results
    return get_pool(status_dir).run(jobs)


def _available_gpus():
    import subprocess

    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {process.stderr}")
    return {int(line) for line in process.stdout.splitlines() if line.strip()}
