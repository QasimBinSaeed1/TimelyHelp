import copy
import json
import numpy as np
import pytest
from timelyhelp.types import Candidate, Control, SimConfig, Message, CameraFrame, Track
from timelyhelp.simulation import (
    ReferenceSimulator,
    make_scenario,
    benchmark_scenarios,
    FEATURE_NAMES,
    FAMILIES,
    WITHHELD_COMPOSITIONS,
)
from timelyhelp.counterfactual import CounterfactualRunner
from timelyhelp.perception import render_rgbd, ColorDepthDetector, ObservationTracker
from timelyhelp.control import EvidenceAdapter


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("setting", ["worksite", "sidewalk"])
def test_all_event_families_and_mobility_settings(family, setting):
    env = ReferenceSimulator(
        make_scenario(family=family, setting=setting), seed=2, config=SimConfig(horizon=2.0)
    )
    for _ in range(6):
        env.advance()
    assert env.history_features().shape == (8, 61)
    assert env.history_features().dtype == np.float32
    assert len(FEATURE_NAMES) == 61
    assert np.isfinite(env.history_features()).all()
    assert set(env.observe()["sources"]) == {"infrastructure", "supervisor"}


def test_frozen_features_preserve_admitted_intent_and_track_dynamics():
    cross = ReferenceSimulator(seed=3)
    clear = cross.fork()
    for env, answer in ((cross, "cross"), (clear, "not_cross")):
        message = Message(
            1,
            "supervisor",
            "will_you_cross_delivery_lane",
            0.0,
            0.0,
            6.0,
            {"answer": answer, "crossing": list(env.geometry["crossings"][0])},
            100,
        )
        assert env.evidence.admit(message, env.tracker, env.time)[0]
    cross_features, clear_features = cross._feature(), clear._feature()
    np.testing.assert_array_equal(cross_features[:32], clear_features[:32])
    assert cross_features[FEATURE_NAMES.index("admitted_intent_cross")] == 1.0
    assert clear_features[FEATURE_NAMES.index("admitted_intent_clear")] == 1.0
    assert not np.array_equal(cross_features, clear_features)
    for env, velocity in ((cross, 0.5), (clear, -0.5)):
        env.evidence.intent_beliefs = []
        env.tracker.tracks = [
            Track(
                np.array([2.0, 0.5]),
                np.array([0.0, velocity]),
                0.1,
                env.time,
                "worker",
                "infrastructure",
            )
        ]
    first, second = cross._feature(), clear._feature()
    np.testing.assert_array_equal(first[:32], second[:32])
    index = FEATURE_NAMES.index("track_1_lateral_velocity")
    assert first[index] == pytest.approx(0.5) and second[index] == pytest.approx(-0.5)
    assert first[FEATURE_NAMES.index("track_1_source")] == 2.0
    # Internal actor changes cannot alter even the newly added features.
    cross.actors[0]["destination"][:] = 100.0
    cross.actors[0]["crosses"] = not cross.actors[0]["crosses"]
    cross.hidden_source_error = {"infrastructure": 1.0, "supervisor": 1.0}
    np.testing.assert_array_equal(first, cross._feature())


def test_observation_does_not_read_hidden_truth():
    env = ReferenceSimulator(seed=3)
    observation = copy.deepcopy(env.observe())
    history = env.history_features().copy()
    env.actors[0]["destination"][:] = 100.0
    env.actors[0]["crosses"] = not env.actors[0]["crosses"]
    env.hidden_source_error = {"infrastructure": 1.0, "supervisor": 1.0}
    assert env.observe() == observation
    np.testing.assert_array_equal(env.history_features(), history)
    encoded = json.dumps(observation)
    for key in ["destination", "corrupt", "actor_id", "hidden_source_error"]:
        assert key not in encoded


def test_collision_at_goal_is_failed_completion_and_retains_failure_cost():
    env = ReferenceSimulator(seed=3, config=SimConfig(horizon=1.0))
    env.goal = env.pose[:2].copy()
    env.actors[0]["position"] = env.pose[:2].copy()
    env.actors[0]["destination"] = env.actors[0]["position"].copy()
    env.step(Control())
    outcome = env.outcomes()
    assert env.done and outcome["collision"] and not outcome["completion"]
    assert outcome["time"] == pytest.approx(0.1)
    assert outcome["cost"] == pytest.approx(10.0 + 2.0 + 0.1 * 0.1 / 1.0)
    from timelyhelp.learning.data import restricted_completion_time

    assert (
        restricted_completion_time({"outcomes": {"A": outcome}, "mission_horizon": 1.0}, "A") == 1.0
    )


def test_rgbd_occlusion_detector_and_projected_native_path():
    actor = {"position": np.array([3.0, 0.0]), "kind": "worker", "radius": 0.26}
    camera = np.array([0.0, 0.0, 0.0])
    detector = ColorDepthDetector()
    visible = render_rgbd(camera, [], [actor], 0)
    assert len(detector.detect(visible)) == 1
    hidden = render_rgbd(camera, [(1.0, -1.0, 2.0, 1.0)], [actor], 0)
    assert detector.detect(hidden) == []
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2:6, 2:6] = [230, 70, 45]
    xy = np.full((8, 8, 2), np.nan)
    xy[2:6, 2:6] = [3.0, 0.0]
    frame = CameraFrame(rgb, np.ones((8, 8)), camera, 0.0, xy_map=xy)
    assert len(detector.detect(frame)) == 1


def test_full_restore_and_reset_replay_include_pending_sensors_controls():
    env = ReferenceSimulator(seed=4, config=SimConfig(horizon=3.0))
    env.request(Candidate("supervisor", "hold", 2.0), exact_delay=2.0)
    for _ in range(8):
        env.advance("cautious")
    snapshot = env.snapshot()
    runner = CounterfactualRunner(env)
    audit = runner.audit_restore(snapshot, steps=8)
    assert audit["passed"], audit
    new = env.fork(snapshot)
    assert new.pending[0].public() == env.pending[0].public()
    np.testing.assert_array_equal(new.history_features(), env.history_features())
    assert new.control_log == env.control_log
    assert new.bridge_log == env.bridge_log
    assert len(new.source_frames) == len(env.source_frames)


@pytest.mark.parametrize("source", ["none", "infrastructure", "supervisor"])
@pytest.mark.parametrize("bridge", ["cautious", "hold"])
@pytest.mark.parametrize("delay", [0.0, 0.25, 0.75])
def test_controlled_branches_and_algebra(source, bridge, delay):
    env = ReferenceSimulator(seed=1, config=SimConfig(horizon=1.5))
    for _ in range(3):
        env.advance()
    row = CounterfactualRunner(env).run(
        env.snapshot(), Candidate(source, bridge, delay), replication=2
    )
    assert row["prearrival_audit"]["passed"]
    assert row["info"] == pytest.approx(row["costs"]["W"] - row["costs"]["Q"])
    assert row["motion"] == pytest.approx(row["costs"]["A"] - row["costs"]["W"])
    assert row["net"] == pytest.approx(
        row["motion"] + row["info"] - env.config.query_cost * (source != "none")
    )
    if source == "none":
        assert row["info"] == 0.0
    assert row["outcomes"]["Q"]["completion_time"] == row["outcomes"]["Q"]["time"]
    json.dumps(row, allow_nan=False)


def test_reactive_actors_use_common_draws_but_actual_robot_motion():
    env = ReferenceSimulator(seed=8)
    env.pose[:2] = env.actors[0]["position"] + np.array([0.9, 0.0])
    a = env.fork()
    b = env.fork()
    assert a.draw("actor_heading", 0) == b.draw("actor_heading", 0)
    a.step(Control(1.0, 0.0))
    b.step(Control(0.0, 0.0))
    assert not np.array_equal(a.actors[0]["position"], b.actors[0]["position"])


def test_sources_unknown_capture_arrival_cancellation_and_expiry():
    env = ReferenceSimulator(
        make_scenario(family="cart_junction"),
        seed=1,
        config=SimConfig(horizon=3.0, request_expiry=0.3),
    )
    worker_id = env.request(Candidate("supervisor", "hold", 0.1), exact_delay=0.1)
    env.advance("hold")
    answer = next(m for m in env.pending if m.request_id == worker_id)
    assert answer.payload["answer"] == "unknown" and answer.reason == "unknown"
    assert answer.capture_time == 0.0 and answer.arrival_time == pytest.approx(0.1)
    request = env.request(Candidate("infrastructure", "hold", 1.0), exact_delay=1.0)
    assert env.cancel(request)
    for _ in range(5):
        env.advance("hold")
    assert next(m for m in env.pending if m.request_id == request).reason == "cancelled"
    late = env.request(Candidate("supervisor", "hold", 2.0), exact_delay=2.0)
    for _ in range(5):
        env.advance("hold")
    assert next(m for m in env.pending if m.request_id == late).reason == "expired"


def test_freshness_rejects_stale_and_newer_belief():
    config = SimConfig()
    adapter = EvidenceAdapter(config)
    tracker = ObservationTracker()
    payload = {"answer": "cross", "crossing": [5.0, 0.0]}
    newer = Message(1, "supervisor", "will_you_cross_delivery_lane", 1.0, 1.1, 6.0, payload, 20)
    assert adapter.admit(newer, tracker, 1.1)[0]
    old = Message(2, "supervisor", "will_you_cross_delivery_lane", 0.5, 1.2, 6.0, payload, 20)
    assert adapter.admit(old, tracker, 1.2) == (False, "newer_source_belief")
    stale = Message(3, "supervisor", "will_you_cross_delivery_lane", 1.0, 4.0, 6.0, payload, 20)
    assert adapter.admit(stale, tracker, 4.0) == (False, "stale")


def test_failed_timeouts_retained_with_exact_cost():
    env = ReferenceSimulator(config=SimConfig(horizon=0.2))
    while not env.done:
        env.advance("hold")
    metrics = env.outcomes()
    assert not metrics["completion"] and not metrics["collision"]
    assert metrics["cost"] == pytest.approx(2.1)
    assert metrics["completion_time"] == pytest.approx(0.2)


def test_layout_split_disjointness_and_withheld_compositions():
    layouts = {
        split: {s.layout_id for s in benchmark_scenarios(split, variants=1)}
        for split in ("train", "development", "calibration", "test")
    }
    for a, ids in layouts.items():
        for b, other in layouts.items():
            if a != b:
                assert not ids & other
    for scenario in benchmark_scenarios("train", variants=1):
        assert len(scenario.disturbances) <= 1
    assert any(
        s.disturbances in WITHHELD_COMPOSITIONS for s in benchmark_scenarios("test", variants=1)
    )
    with pytest.raises(ValueError):
        make_scenario("train", disturbances=("dropout", "packet_loss"))


def test_response_prior_excludes_pending_and_cancelled_requests():
    env = ReferenceSimulator(config=SimConfig(horizon=1.0, request_expiry=0.3))
    request = env.request(Candidate("supervisor", "hold", 0.5), exact_delay=0.5)
    assert env.observe()["sources"]["supervisor"]["response_fraction"] == 1.0
    env.cancel(request)
    assert env.observe()["sources"]["supervisor"]["response_fraction"] == 1.0
    env.request(Candidate("supervisor", "hold", 0.5), exact_delay=0.5)
    for _ in range(4):
        env.advance("hold")
    assert env.observe()["sources"]["supervisor"]["response_fraction"] == pytest.approx(0.5)


def test_speed_and_source_view_assignment_are_observable_and_not_source_labels():
    source_fast = []
    coverages = []
    for seed in range(8):
        env = ReferenceSimulator(seed=seed)
        source = env.observe()["sources"]
        coverages.append(source["infrastructure"]["coverage"])
        fast = min(source, key=lambda key: np.mean(source[key]["latencies"]))
        source_fast.append(fast)
        assert source["infrastructure"]["latencies"] == env.advertised_latency["infrastructure"]
    assert set(source_fast) == {"infrastructure", "supervisor"}
    assert len(set(coverages)) == 2


def test_second_controller_is_distinct_and_preserves_safety():
    from timelyhelp.control import make_planner

    env = ReferenceSimulator(seed=0)
    obs = env.observe()
    obs["blind_zone_uncertainty"] = 1.0
    first = make_planner(SimConfig(controller="local")).plan(obs)
    second = make_planner(SimConfig(controller="reactive")).plan(obs)
    assert second.linear < first.linear
    obs["depth_min"] = 0.01
    assert make_planner(SimConfig(controller="local")).plan(obs).linear == 0.0
    assert make_planner(SimConfig(controller="reactive")).plan(obs).linear == 0.0


def test_source_velocities_and_stale_onboard_protection():
    tracker = ObservationTracker()
    detection = {
        "position": np.array([3.0, 0.0]),
        "velocity": [0.2, -0.8],
        "covariance": 0.02,
        "capture_time": 1.0,
        "kind": "worker",
    }
    tracker.update([detection], 1.0, provenance="infrastructure")
    np.testing.assert_allclose(tracker.tracks[0].velocity, [0.2, -0.8])
    older = {
        **detection,
        "capture_time": 0.8,
        "position": np.array([3.1, 0.0]),
        "velocity": [2.0, 2.0],
    }
    tracker.update([older], 1.1, provenance="infrastructure")
    np.testing.assert_allclose(tracker.tracks[0].velocity, [0.2, -0.8])
    assert tracker.tracks[0].capture_time == 1.0


def test_packet_reassessment_uses_new_onboard_capture_before_fusion():
    env = ReferenceSimulator(seed=0, config=SimConfig(horizon=0.3))
    values = []

    def admit(packet, observation):
        values.append((env.history_features()[-1, 0], env.current_history_features()[-1, 0]))
        return True

    env.admission_filter = admit
    env.request(Candidate("infrastructure", "hold", 0.1), exact_delay=0.1)
    env.advance("hold")
    assert values[0][0] == pytest.approx(0.0)
    assert values[0][1] == pytest.approx(1.0 / 3)
    assert len(env.history) == 2
