"""Local browser diagnostics for bounded native missions, isolated per GPU.

Only localhost is served. SSH port forwarding carries the ordinary HTTP frames
and controls, so no WebRTC/UDP route or desktop forwarding is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def parse_gpus(value):
    try:
        indices = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from error
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must be unique nonnegative integers")
    return indices


class DemoSession:
    def __init__(self, output, gpus, duration, minutes):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        previous = read_json(self.output / "session.json", {})
        if isinstance(previous.get("pid"), int):
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("This output directory belongs to an active viewer; stop it first")
        self.gpus = tuple(gpus)
        self.duration = duration
        self.expires = time.time() + minutes * 60
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.control = {
            "run": 0,
            "paused": False,
            "stop": False,
            "view": "follow",
            "azimuth": -55,
            "elevation": 34,
            "distance": 12,
        }
        self.processes = {}
        self.logs = []
        for gpu in self.gpus:
            directory = self.output / f"gpu_{gpu}"
            directory.mkdir(exist_ok=True)
            # Reusing a directory must not expose previous-session ready states.
            atomic_json(directory / "status.json", {"gpu": gpu, "phase": "starting"})
        atomic_json(self.output / "control.json", self.control)

    def launch(self):
        from .config import provenance

        record = provenance(
            {
                "mode": "smoke",
                "backend": "isaac",
                "gpus": self.gpus,
                "duration": self.duration,
                "kind": "interactive_native_demo",
            },
            stage="live_viewer",
        )
        record["viewer_html_sha256"] = hashlib.sha256(
            Path(__file__).with_name("assets").joinpath("live.html").read_bytes()
        ).hexdigest()
        atomic_json(self.output / "provenance.json", record)
        for slot, gpu in enumerate(self.gpus):
            log = (self.output / f"gpu_{gpu}.log").open("w")
            self.logs.append(log)
            environment = dict(
                os.environ,
                TIMELYHELP_GPU=str(gpu),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONUNBUFFERED="1",
            )
            self.processes[gpu] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "timelyhelp.live_worker",
                    "--gpu",
                    str(gpu),
                    "--slot",
                    str(slot),
                    "--duration",
                    str(self.duration),
                    "--output",
                    str(self.output),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(
            self.output / "session.json",
            {
                "kind": "interactive_native_demo",
                "backend": "isaac",
                "gpus": self.gpus,
                "duration_per_mission_seconds": self.duration,
                "expires_unix": self.expires,
                "pid": os.getpid(),
                "worker_pids": {gpu: p.pid for gpu, p in self.processes.items()},
                "training_executed": False,
                "full_pilot_executed": False,
                "policy": "always_query baseline; no trained-model performance claim",
                "camera": "human-only overview plus actual onboard/infrastructure RGB-D",
            },
        )

    def status(self):
        jobs = []
        for gpu in self.gpus:
            row = read_json(
                self.output / f"gpu_{gpu}/status.json", {"gpu": gpu, "phase": "starting"}
            )
            process = self.processes.get(gpu)
            if process is not None and process.poll() is not None and row["phase"] != "error":
                row = dict(row, phase="stopped", worker_exit_code=process.returncode)
            jobs.append(row)
        with self.lock:
            control = dict(self.control)
        return {
            "jobs": jobs,
            "control": control,
            "duration": self.duration,
            "expires_unix": self.expires,
            "server_pid": os.getpid(),
        }

    def command(self, action, value=None):
        # One shared control file makes a run request atomic across all workers.
        with self.lock:
            if action == "run":
                rows = [read_json(self.output / f"gpu_{gpu}/status.json", {}) for gpu in self.gpus]
                if any(row.get("phase") not in ("ready", "done") for row in rows):
                    raise ValueError("Wait until all instances are ready or finished")
                if any(row.get("run", -1) != self.control["run"] for row in rows):
                    raise ValueError("The previous run request is still being accepted")
                self.control["run"] += 1
                self.control["paused"] = False
            elif action == "pause":
                self.control["paused"] = not self.control["paused"]
            elif action == "view" and value in ("overview", "top", "follow", "orbit"):
                self.control["view"] = value
            elif action == "orbit" and isinstance(value, dict):
                values = {key: float(value[key]) for key in ("azimuth", "elevation", "distance")}
                if not all(math.isfinite(number) for number in values.values()):
                    raise ValueError("Camera parameters must be finite")
                self.control.update(
                    view="orbit",
                    azimuth=values["azimuth"] % 360,
                    elevation=min(85, max(10, values["elevation"])),
                    distance=min(30, max(2, values["distance"])),
                )
            elif action == "stop":
                self.control["stop"] = True
                self.stop.set()
            else:
                raise ValueError("Unknown demo command or camera view")
            atomic_json(self.output / "control.json", self.control)
            return dict(self.control)

    def close(self):
        self.command("stop")
        deadline = time.monotonic() + 15
        for process in self.processes.values():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                # Own process group includes the isolated Isaac subprocess.
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in self.logs:
            log.close()


def make_server(session, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, content, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_reply(self, code, value):
            self.reply(code, json.dumps(value).encode(), "application/json")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.reply(
                    200,
                    Path(__file__).with_name("assets").joinpath("live.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/api/status":
                self.json_reply(200, session.status())
            elif path.startswith("/frame/"):
                parts = path.split("/")
                if (
                    len(parts) != 4
                    or parts[2] not in {str(gpu) for gpu in session.gpus}
                    or parts[3]
                    not in {"world.jpg", "onboard.jpg", "infrastructure.jpg", "depth.jpg"}
                ):
                    self.json_reply(404, {"error": "Unknown frame"})
                    return
                frame = session.output / f"gpu_{parts[2]}" / parts[3]
                try:
                    self.reply(200, frame.read_bytes(), "image/jpeg")
                except FileNotFoundError:
                    self.json_reply(503, {"error": "Camera is warming up"})
            else:
                self.json_reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/control":
                self.json_reply(404, {"error": "Unknown endpoint"})
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                self.json_reply(403, {"error": "Cross-origin controls are disabled"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024:
                    raise ValueError("Invalid request size")
                command = json.loads(self.rfile.read(length))
                self.json_reply(200, session.command(command["action"], command.get("value")))
            except (ValueError, KeyError, TypeError) as error:
                self.json_reply(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.timeout = 0.25
    return server


def serve_live(gpus, port, output, duration=18, session_minutes=60):
    if not 0 < duration <= 30 or not 0 < session_minutes <= 720:
        raise ValueError("Demo duration must be 0–30 seconds; session lifetime 0–720 minutes")
    if not 0 <= port <= 65535:
        raise ValueError("Invalid port")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    available = {int(line) for line in result.stdout.splitlines()}
    if not set(gpus) <= available:
        raise ValueError(f"Requested GPUs unavailable; available indices: {sorted(available)}")
    session = DemoSession(output, gpus, duration, session_minutes)
    server = make_server(session, port)
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: session.stop.set())
    try:
        session.launch()
        print(f"Live viewer: http://127.0.0.1:{server.server_port} (PID {os.getpid()})", flush=True)
        print(
            "Waiting for browser Run command. Each click runs one bounded mission per GPU.",
            flush=True,
        )
        while not session.stop.is_set() and time.time() < session.expires:
            server.handle_request()
    finally:
        session.close()
        server.server_close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
