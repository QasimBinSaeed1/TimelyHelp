"""Diagnostics must reject incomplete traces rather than passing an empty zip."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_native_pairs import (
    compare_samples,
    csv_values,
    stationary_clock_check,
    trace_contract,
)
from timelyhelp.counterfactual import CounterfactualRunner
from timelyhelp.types import Control


def result(times, *, done=False, collision=False):
    return {
        "trace": [{"time": time} for time in times],
        "end_time": times[-1] if times else 0.5,
        "done": done,
        "collision": collision,
        "completion": False,
    }


def test_prearrival_contract_includes_tick_straddling_arrival():
    branch = result([0.6, 0.7, 0.8])
    assert trace_contract(branch, branch, 0.5, 0.25, 0.1, 25.0)["passed"]
    short = result([0.6, 0.7])
    assert not trace_contract(short, short, 0.5, 0.25, 0.1, 25.0)["passed"]


def test_prearrival_contract_rejects_empty_or_unequal_traces():
    empty = result([])
    full = result([0.6, 0.7, 0.8])
    assert not trace_contract(empty, empty, 0.5, 0.25, 0.1, 25.0)["passed"]
    assert not trace_contract(empty, full, 0.5, 0.25, 0.1, 25.0)["passed"]


def test_prearrival_contract_checks_actual_timestamps_and_termination():
    first = result([0.6, 0.7, 0.8])
    reordered = result([0.7, 0.6, 0.8])
    assert not trace_contract(first, reordered, 0.5, 0.25, 0.1, 25.0)["passed"]
    premature = result([0.6], done=True)
    assert not trace_contract(premature, premature, 0.5, 0.25, 0.1, 25.0)["passed"]
    collision = result([0.6], done=True, collision=True)
    assert trace_contract(collision, collision, 0.5, 0.25, 0.1, 25.0)["passed"]
    assert not trace_contract(collision, result([0.6]), 0.5, 0.25, 0.1, 25.0)["passed"]


def test_sample_comparison_checks_prefix_rgb_control_pose_and_time():
    runner = CounterfactualRunner(SimpleNamespace())
    sample = {
        "time": 0.5,
        "pose": np.zeros(3),
        "control": Control(),
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth": np.ones((2, 3)),
        "dropped": False,
    }
    assert compare_samples(runner, sample, sample)["passed"]
    for key in ("rgb", "pose", "control", "time"):
        changed = deepcopy(sample)
        if key == "control":
            changed[key].angular = 0.01
        else:
            changed[key] += 1
        assert not compare_samples(runner, sample, changed)["passed"]


def test_stationary_native_clock_accepts_a_different_constant_origin():
    native = {label: 0.4999999888 for label in ("first", "repeated", "interleaved")}
    wrapper = dict.fromkeys(native, 0.5)
    check = stationary_clock_check(native, wrapper)
    assert check["passed"]
    assert check["native_clock_change_max_s"] == 0
    assert check["native_minus_wrapper_offsets_s"]["first"] != 0
    assert check["native_capture_times"] == native


def test_stationary_native_clock_rejects_actual_capture_time_advance():
    native = {"first": 0.4999999888, "interleaved": 0.5999999888}
    wrapper = dict.fromkeys(native, 0.5)
    check = stationary_clock_check(native, wrapper)
    assert not check["passed"]
    assert check["native_clock_change_max_s"] == pytest.approx(0.1)
    assert check["wrapper_clock_change_max_s"] == 0


@pytest.mark.parametrize("value", ["", "0,0", "0,,1"])
def test_probe_lists_cannot_silently_drop_all_or_duplicate_cases(value):
    with pytest.raises(ValueError):
        csv_values(value, int)
