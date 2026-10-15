"""Read-only pilot monitor: watch -n 10 .conda/bin/python scripts/pilot_status.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


JOB_LABELS = {
    "generate_episode": "generation episodes",
    "calibration_episode": "calibration missions",
    "mission": "evaluation missions",
    "oracle": "privileged oracle missions",
    "profile": "profiling missions",
}
PREFIX_KINDS = {"generate": "generate_episode", "calibration": "calibration_episode"}


def pid_is_alive(pid):
    """Check the actual process; a zombie is not an active worker."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except FileNotFoundError:
        return False
    except (PermissionError, IndexError):
        return True
    return state not in {"Z", "X"}


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"read_error": "expected a JSON object"}
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error)}


def _age(stamp, now):
    if not isinstance(stamp, (int, float)):
        return "age unavailable"
    elapsed = now - stamp
    if elapsed < 0:
        return "timestamp ahead of monitor clock"
    seconds = int(elapsed)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s ago"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m ago"


def _state(record, *, pid=None, worker=False):
    recorded = record.get("status") or record.get("stage") or "unknown"
    pid = record.get("pid") if pid is None else pid
    alive = pid_is_alive(pid)
    process = f"PID {pid} {'alive' if alive else 'dead'}" if pid else "PID unavailable"
    if recorded in {"failed", "interrupted", "stopped"}:
        return f"{recorded.upper()} ({process})"
    if recorded == "complete":
        if worker:
            return f"{'IDLE' if alive else 'STOPPED'}; last job complete ({process})"
        return f"COMPLETE ({process})"
    if pid and not alive:
        return f"STOPPED/stale (recorded {recorded}; {process})"
    if alive:
        return f"{recorded.upper()} ({process})"
    return f"UNKNOWN ({process}; recorded {recorded})"


def format_status(output, *, now=None):
    """Format only recorded evidence and independently verified PID liveness."""
    output = Path(output)
    now = time.time() if now is None else now
    run = _read(output / "run_status.json")
    pool = _read(output / "workers/progress.json")
    workers = [(path, _read(path)) for path in sorted((output / "workers").glob("gpu_*.json"))]
    lines = [f"TimelyHelp pilot: {output}"]
    if run is None:
        lines.append("Pipeline: no run_status.json; no running/completion claim is available.")
        run = {}
    elif "read_error" in run:
        lines.append(f"Pipeline status unreadable: {run['read_error']}")
    else:
        lines.append(
            f"Pipeline: {_state(run)} | stage={run.get('stage', 'unknown')} | updated {_age(run.get('unix_time'), now)}"
        )
        if run.get("resumable_stage"):
            lines.append(f"Resumable stage: {run['resumable_stage']}")
        if run.get("error"):
            error = run["error"]
            lines.append(
                f"Pipeline error: {error.get('type', '')}: {error.get('message', '')}"
                if isinstance(error, dict)
                else f"Pipeline error: {error}"
            )
    if pool is not None and "read_error" not in pool:
        kinds = pool.get("job_kinds") or []
        if not kinds:
            job_ids = list(pool.get("inflight", {}).values())
            job_ids += [record.get("job") for _, record in workers if record]
            prefixes = {job.split("/", 1)[0] for job in job_ids if isinstance(job, str)}
            kinds = sorted(PREFIX_KINDS.get(prefix, prefix) for prefix in prefixes)
        label = (
            ", ".join(JOB_LABELS.get(kind, kind) for kind in kinds)
            or "simulation jobs (kind unavailable)"
        )
        coordinator = pool.get("pid") or run.get("pid")
        lines.append(
            f"Simulation schedule ({label}): {_state(pool, pid=coordinator)} | {pool.get('completed', '?')}/{pool.get('total', '?')} completed | failures={pool.get('failures', '?')} | heartbeat {_age(pool.get('unix_time'), now)}"
        )
        if pool.get("failed_job_ids"):
            lines.append("Failed jobs: " + ", ".join(pool["failed_job_ids"]))
    elif pool is not None:
        lines.append(f"Simulation status unreadable: {pool['read_error']}")
    else:
        lines.append("Simulation schedule: no worker progress file yet.")
    for path, record in workers:
        if not record or "read_error" in record:
            lines.append(f"{path.stem}: status unavailable or unreadable")
            continue
        label = _state(record, worker=True)
        lines.append(
            f"GPU {record.get('gpu', path.stem.removeprefix('gpu_'))}: {label} | heartbeat {_age(record.get('unix_time'), now)}"
        )
        if record.get("job"):
            lines.append(f"  Recorded job: {record['job']}")
        elif record.get("last_job"):
            lines.append(f"  Last job: {record['last_job']}")
        if record.get("job_progress"):
            lines.append(f"  Branch progress: {record['job_progress']}")
        if record.get("message"):
            lines.append(f"  Latest update: {record['message']}")
        if record.get("error"):
            lines.append(f"  Worker error: {str(record['error']).splitlines()[-1]}")
    model = _read(output / "models/progress.json")
    if model and "read_error" not in model:
        lines.append(
            "Last completed model/calibration: "
            + ", ".join(f"{key}={value}" for key, value in model.items())
        )
    elif model:
        lines.append(f"Model progress unreadable: {model['read_error']}")
    lines.append(
        "The counter counts completed whole jobs. During generation, one episode can run many branches while its episode counter stays unchanged; compare branch progress and physics updates between refreshes."
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/pilot")
    args = parser.parse_args(argv)
    print(format_status(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
