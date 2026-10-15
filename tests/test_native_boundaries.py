"""Native sensor schema and physical-contact coverage without loading an engine."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import numpy as np

from timelyhelp.backends.worker_common import foreign_physics_contact
from timelyhelp.backends.worlds import gazebo_world, camera_xy_map
from timelyhelp.backends.native import (
    _native_actor_velocity,
    NATIVE_ACTOR_BEHAVIOR,
    NativeSimulator,
)
from timelyhelp.simulation import make_scenario
from timelyhelp.types import SimConfig, Control


def _contact(first, second):
    return SimpleNamespace(
        collision1=SimpleNamespace(name=first), collision2=SimpleNamespace(name=second)
    )


def test_native_contact_protobuf_entities_and_label_filter():
    assert foreign_physics_contact(_contact("robot::chassis::body", "actor_0::body::actor"))
    assert foreign_physics_contact(_contact("stack_0::link::box", "robot::left_wheel::wheel"))
    assert not foreign_physics_contact(
        _contact("robot::rear_caster::caster", "ground::link::ground")
    )
    assert not foreign_physics_contact(_contact("robot::chassis::body", "robot::payload::load"))
    with pytest.raises(ValueError, match="omitted"):
        foreign_physics_contact(_contact("", "robot::chassis::body"))
    with pytest.raises(ValueError, match="without the robot"):
        foreign_physics_contact(_contact("actor_0::body::actor", "ground::link::ground"))


def test_gazebo_all_robot_colliders_have_native_contact_sensors():
    config = SimConfig(camera_height=24, camera_width=80).to_dict()
    world = ET.fromstring(gazebo_world(make_scenario().geometry(), [], config))
    robot = world.find("./world/model[@name='robot']")
    covered = set()
    for link in robot.findall("link"):
        for sensor in link.findall("sensor[@type='contact']"):
            contact = sensor.find("contact")
            assert contact.findtext("topic") in ("/robot_contacts", "/payload_contacts")
            covered.add((link.attrib["name"], contact.findtext("collision")))
    physical = {
        (link.attrib["name"], collision.attrib["name"])
        for link in robot.findall("link")
        for collision in link.findall("collision")
    }
    assert physical == covered
    for image in world.findall(".//sensor[@type='rgbd_camera']/camera/image"):
        assert int(image.findtext("height")) == config["camera_height"]
        assert int(image.findtext("width")) == config["camera_width"]


def test_native_missing_depth_rays_are_unavailable_without_invalid_arithmetic():
    depth = np.array([[np.inf, np.nan, 0.0], [-1.0, 2.0, 3.0]], dtype=np.float32)
    with np.errstate(all="raise"):
        xy = camera_xy_map(depth, [1.0, 2.0, 0.2], 0.85, 0.0, 2.2)
    assert np.isnan(xy[0]).all()
    assert np.isnan(xy[1, 0]).all()
    assert np.isfinite(xy[1, 1:]).all()


def test_backprojection_uses_native_camera_intrinsics_and_full_extrinsics():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    calibration = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])
    xy = camera_xy_map(depth, [10.0, 20.0, 0.0], 3.0, 0.0, 2.2, calibration)
    np.testing.assert_allclose(xy[1, 1], [12.0, 20.0])
    np.testing.assert_allclose(xy[1, 0], [12.0, 21.0])
    downward = camera_xy_map(depth, [10.0, 20.0, 0.0], 3.0, np.pi / 2, 2.2, calibration)
    np.testing.assert_allclose(downward[0, 1], [10.5, 20.0], atol=1e-6)
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    full = camera_xy_map(
        depth, [10.0, 20.0, 0.0], 3.0, 0.0, 2.2, calibration, rotation, [10.0, 20.0, 3.0]
    )
    np.testing.assert_allclose(full, downward, atol=1e-6)
    invalid = calibration.copy()
    invalid[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive focal"):
        camera_xy_map(depth, [10.0, 20.0, 0.0], 3.0, 0.0, 2.2, invalid)


def test_gazebo_transfer_actor_shift_preserves_common_noise_and_reactivity():
    actor = {"position": np.array([5.0, 0.0]), "destination": np.array([10.0, 0.0]), "speed": 1.0}
    pose = np.zeros(3)
    primary = _native_actor_velocity(actor, pose, 0.5, 0.4, "isaac")
    transfer = _native_actor_velocity(actor, pose, 0.5, 0.4, "gazebo")
    assert transfer[0] / primary[0] == pytest.approx(1.1)
    assert transfer[1] == pytest.approx(primary[1])  # Shared exogenous heading draw.
    nearby = np.array([4.5, 0.0, 0.0])
    transfer_near = _native_actor_velocity(actor, nearby, 0.5, 0.4, "gazebo")
    assert not np.allclose(transfer, transfer_near)  # Reacts to the actual robot pose.
    assert NATIVE_ACTOR_BEHAVIOR["isaac"]["moving_robot_yield_factor"] == 0.75
    actor["destination"] = actor["position"].copy()
    assert np.array_equal(_native_actor_velocity(actor, nearby, 0.5, 0.4, "gazebo"), np.zeros(2))


@pytest.mark.parametrize("collision_phase", ["physics", "render"])
def test_native_goal_contact_is_failure_including_deferred_contact_callback(
    tmp_path, collision_phase
):
    """Test native RPC plumbing with synthetic replies, not an engine claim."""
    env = object.__new__(NativeSimulator)
    env.config = SimConfig(horizon=1.0)
    env.pose = np.array([0.0, 0.0, 0.0])
    env.goal = np.array([1.0, 0.0])
    env.actors = []
    env.scenario = SimpleNamespace(disturbances=())
    env.geometry = {"rectangles": []}
    env.velocity = env.angular_velocity = env.distance = env.time = 0.0
    env.tick = env.near_misses = env.queries = env.communication_bytes = env.workload = 0
    env._near_contact = env.collision = env.completion = env.done = False
    env.reply_delays = []
    env._ensure_engine = lambda: None
    frame_path = tmp_path / "frame.npz"
    np.savez(
        frame_path,
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        depth=np.ones((1, 1), dtype=np.float32),
        xy_map=np.zeros((1, 1, 2)),
    )

    def rpc(command, data):
        if command == "step":
            return {
                "pose": [1.0, 0.0, 0.0],
                "velocity": [0.0, 0.0],
                "collision": collision_phase == "physics",
            }
        return {
            "pose": [1.0, 0.0, 0.0],
            "path": str(frame_path),
            "capture_time": 0.1,
            "collision": collision_phase == "render",
        }

    env._driver = SimpleNamespace(rpc=rpc)
    env._physics_step(Control(0.0, 0.0))
    if collision_phase == "render":
        env._render(env.pose)
    outcome = env.outcomes()
    assert outcome["collision"] and not outcome["completion"] and env.done
    assert outcome["cost"] == pytest.approx(12.01)


def test_native_driver_pool_reuses_parent_and_branch_and_closes_overflow(monkeypatch):
    """Bound idle process ownership without creating native-engine processes."""
    import timelyhelp.backends.native as native

    class StubDriver:
        def __init__(self):
            self.closed = False
            self.close_count = 0
            self.process = SimpleNamespace(poll=lambda: None)

        def close(self):
            self.closed = True
            self.close_count += 1

    pools = {"isaac": [], "gazebo": []}
    monkeypatch.setattr(native, "_DRIVER_POOL", pools)
    drivers = [StubDriver() for _ in range(3)]
    for driver in drivers:
        env = object.__new__(NativeSimulator)
        env._driver = driver
        env._closed = False
        env._native_backend = "gazebo"
        env.close()
        env.close()  # Repeated release cannot duplicate idle ownership.
    assert pools["gazebo"] == drivers[:2]
    assert drivers[2].close_count == 1
    assert not any(driver.closed for driver in drivers[:2])

    monkeypatch.setattr(native.ReferenceSimulator, "__init__", lambda *args: None)
    resumed = NativeSimulator(backend="gazebo")
    assert resumed._driver is drivers[1]
    assert pools["gazebo"] == [drivers[0]]
    resumed.close()
    native._close_pool()
    assert pools == {"isaac": [], "gazebo": []}
    assert all(driver.closed and driver.close_count == 1 for driver in drivers)
