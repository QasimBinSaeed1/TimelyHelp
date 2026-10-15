"""HTTP control contracts and observer neutrality; native checks are separate."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from timelyhelp.baselines import make_baseline
from timelyhelp.backends.presentation import orbit_camera
from timelyhelp.evaluation import run_episode
from timelyhelp.live import DemoSession, atomic_json, make_server, parse_gpus
from timelyhelp.live_worker import depth_pixels, write_jpeg
from timelyhelp.simulation.reference import ReferenceSimulator
from timelyhelp.types import SimConfig


@pytest.mark.parametrize("value", ["", "-1", "0,0", "0,a", "0,1,"])
def test_invalid_gpu_lists(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpus(value)


def test_gpu_order_is_preserved():
    assert parse_gpus("3,1,0,2") == (3, 1, 0, 2)


def test_orbit_camera_geometry_and_finite_validation():
    np.testing.assert_allclose(
        orbit_camera([1, 2, 3], 0, 30, 10), [1 + 10 * np.cos(np.pi / 6), 2, 8]
    )
    eye = orbit_camera([0, 0, 0], -90, -400, 1000)
    assert np.isclose(np.linalg.norm(eye), 30)
    assert eye[2] > 0
    with pytest.raises(ValueError, match="finite"):
        orbit_camera([0, 0, 0], float("nan"))


def test_output_directory_cannot_overwrite_an_active_session(tmp_path):
    atomic_json(tmp_path / "session.json", {"pid": os.getpid()})
    with pytest.raises(ValueError, match="active viewer"):
        DemoSession(tmp_path, (0,), 1, 1)


def test_diagnostic_callback_preserves_episode():
    config = SimConfig(horizon=1, dt=0.1)
    first = ReferenceSimulator(seed=19, config=config)
    second = ReferenceSimulator(seed=19, config=config)
    captured = []

    def observer(env, telemetry):
        captured.append((deepcopy(env.observe()), deepcopy(telemetry)))
        return {"bridge": "hold", "source": "supervisor"}

    expected = run_episode(first, make_baseline("always_query"))
    observed = run_episode(second, make_baseline("always_query"), step_callback=observer)
    for result in (expected, observed):
        result.pop("selection_seconds_mean")
        result.pop("selection_seconds_max")
    assert observed == expected
    assert len(captured) == expected["steps"]
    assert [t[1]["step"] for t in captured] == list(range(1, expected["steps"] + 1))
    assert captured[-1][0] == first.observe()


def test_depth_jpeg_handles_invalid_pixels(tmp_path):
    pixels = depth_pixels(np.array([[0, 7, np.nan, np.inf]]), 7)
    assert pixels.tolist() == [[[255, 220, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255]]]
    write_jpeg(tmp_path / "depth.jpg", pixels)
    assert (tmp_path / "depth.jpg").read_bytes().startswith(b"\xff\xd8")
    assert not (tmp_path / "depth.tmp").exists()


@pytest.fixture
def viewer(tmp_path):
    session = DemoSession(tmp_path, (0, 1), 1, 1)
    server = make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield session, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(url, action=None, origin=None):
    if action is None:
        return urlopen(url, timeout=2)
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    return urlopen(Request(url, data=json.dumps(action).encode(), headers=headers), timeout=2)


def test_http_frames_status_and_route_bounds(viewer):
    session, base = viewer
    assert b"Live native simulations" in request(base + "/").read()
    status = json.load(request(base + "/api/status"))
    assert [job["phase"] for job in status["jobs"]] == ["starting", "starting"]
    with pytest.raises(HTTPError) as error:
        request(base + "/frame/0/world.jpg")
    assert error.value.code == 503
    write_jpeg(session.output / "gpu_0/world.jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    assert request(base + "/frame/0/world.jpg?nonce=1").read().startswith(b"\xff\xd8")
    for path in ("/frame/0/../../control.json", "/frame/2/world.jpg", "/frame/0/status.json"):
        with pytest.raises(HTTPError) as error:
            request(base + path)
        assert error.value.code == 404


def test_start_is_atomic_and_cannot_interrupt_unaccepted_or_running_missions(viewer):
    session, base = viewer
    with pytest.raises(HTTPError) as error:
        request(base + "/api/control", {"action": "run"})
    assert error.value.code == 400
    for gpu in session.gpus:
        atomic_json(
            session.output / f"gpu_{gpu}/status.json", {"gpu": gpu, "phase": "ready", "run": 0}
        )
    assert json.load(request(base + "/api/control", {"action": "run"}))["run"] == 1
    with pytest.raises(HTTPError):
        request(base + "/api/control", {"action": "run"})
    assert session.control["run"] == 1
    for gpu in session.gpus:
        atomic_json(
            session.output / f"gpu_{gpu}/status.json", {"gpu": gpu, "phase": "done", "run": 1}
        )
    assert json.load(request(base + "/api/control", {"action": "run"}))["run"] == 2


def test_controls_pause_view_stop_and_origin(viewer):
    session, base = viewer
    url = base + "/api/control"
    assert json.load(request(url, {"action": "pause"}, base))["paused"]
    assert not json.load(request(url, {"action": "pause"}))["paused"]
    assert json.load(request(url, {"action": "view", "value": "top"}))["view"] == "top"
    control = json.load(
        request(
            url, {"action": "orbit", "value": {"azimuth": 400, "elevation": 0, "distance": 100}}
        )
    )
    assert control["view"] == "orbit"
    assert (control["azimuth"], control["elevation"], control["distance"]) == (40, 10, 30)
    for value in ({"azimuth": float("nan"), "elevation": 30, "distance": 10}, {"azimuth": 10}):
        with pytest.raises(HTTPError):
            request(url, {"action": "orbit", "value": value})
    with pytest.raises(HTTPError) as error:
        request(url, {"action": "stop"}, "http://unrelated.example")
    assert error.value.code == 403
    with pytest.raises(HTTPError):
        request(url, {"action": "view", "value": "secret"})
    assert json.load(request(url, {"action": "stop"}))["stop"]
    assert session.stop.is_set()
    assert json.loads(Path(session.output / "control.json").read_text())["stop"]
