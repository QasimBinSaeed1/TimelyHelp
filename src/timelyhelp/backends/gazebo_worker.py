"""Native Gazebo Harmonic physics and RGB-D worker (system Python 3.10)."""

from __future__ import annotations
import math
import threading
import time
import numpy as np
from gz.sim8 import TestFixture, World, Model, Link, Joint, world_entity
from gz.math7 import Pose3d
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.camera_info_pb2 import CameraInfo

from worlds import gazebo_world, camera_xy_map
from worker_common import serve, foreign_physics_contact


class GazeboEngine:
    def __init__(self, cache):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fixture = None
        self.node = None
        self.frames = {}
        self.calibrations = {}
        self.collision = False
        self.contact_stamps = {}
        self.contact_counts = {}
        self.lock = threading.Lock()
        self.counter = 0
        self.physics_time = 0.0
        self.transport_errors = []
        self.generation = 0

    def reset(self, data):
        if self.fixture is not None:
            self.close()
        self.config, self.geometry = data["config"], data["geometry"]
        self.actor_states = data["actors"]
        self.control = [0.0, 0.0]
        self.models = {}
        self.frames = {}
        self.calibrations = {}
        self.contact_stamps = {}
        self.contact_counts = {}
        self.transport_errors = []
        self.generation += 1
        generation = self.generation
        self.physics_time = 0.0
        path = self.cache / "world.sdf"
        path.write_text(gazebo_world(self.geometry, self.actor_states, self.config))
        self.fixture = TestFixture(str(path))
        self.fixture.on_pre_update(self._pre_update)
        self.fixture.on_post_update(self._post_update)
        self.fixture.finalize()
        self.node = Node()
        for topic in ("/robot_contacts", "/payload_contacts"):

            def contacts_received(message, topic=topic):
                stamp = message.header.stamp.sec + message.header.stamp.nsec * 1e-9
                with self.lock:
                    if generation != self.generation:
                        return
                    self.contact_stamps[topic] = stamp
                    self.contact_counts[topic] = len(message.contact)
                    for contact in message.contact:
                        # Harmonic Contacts carries Entity messages, not strings.
                        # Read the physics-generated scoped names; treating the
                        # protobuf objects as names raises in transport callbacks
                        # and silently loses the collision label.
                        try:
                            if foreign_physics_contact(contact):
                                self.collision = True
                        except (ValueError, AttributeError) as exc:
                            self.transport_errors.append(str(exc))

            if not self.node.subscribe(Contacts, topic, contacts_received):
                raise RuntimeError(f"Cannot subscribe native physics contacts {topic}")
        for name in ("onboard", "infrastructure"):

            def calibration_received(message, source=name):
                matrix = np.asarray(message.intrinsics.k, dtype=float)
                with self.lock:
                    if generation != self.generation:
                        return
                    if matrix.size != 9 or not np.isfinite(matrix).all():
                        self.transport_errors.append(
                            f"Invalid native camera calibration for {source}"
                        )
                    else:
                        self.calibrations[source] = matrix.reshape(3, 3)

            if not self.node.subscribe(CameraInfo, f"/{name}/camera_info", calibration_received):
                raise RuntimeError(f"Cannot subscribe native calibration {name}/camera_info")
            for kind, suffix in (("rgb", "image"), ("depth", "depth_image")):

                def receive(msg, key=(name, kind)):
                    stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
                    if key[1] == "rgb":
                        array = (
                            np.frombuffer(msg.data, np.uint8)
                            .reshape(msg.height, msg.width, -1)[..., :3]
                            .copy()
                        )
                    else:
                        array = (
                            np.frombuffer(msg.data, np.float32)
                            .reshape(msg.height, msg.width)
                            .copy()
                        )
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.frames[key] = (stamp, array)

                if not self.node.subscribe(Image, f"/{name}/{suffix}", receive):
                    raise RuntimeError(f"Cannot subscribe native camera {name}/{suffix}")
        # Settle contacts before defining the common-prefix episode time origin.
        self.fixture.server().run(True, 50, False)
        self.origin = self.physics_time
        self.collision = False
        return {
            "pose": self.pose,
            "backend": "gazebo",
            "physics_time": self.physics_time,
            "collision": False,
        }

    def _pre_update(self, info, ecm):
        world = World(world_entity(ecm))
        if not self.models:
            self.models["robot"] = Model(world.model_by_name(ecm, "robot"))
            self.robot_link = Link(self.models["robot"].link_by_name(ecm, "chassis"))
            self.robot_link.enable_velocity_checks(ecm)
            self.wheels = [
                Joint(self.models["robot"].joint_by_name(ecm, f"{side}_joint"))
                for side in ("left", "right")
            ]
            for i in range(len(self.actor_states)):
                self.models[f"actor_{i}"] = Model(world.model_by_name(ecm, f"actor_{i}"))
        linear, angular = self.control
        speeds = [(linear - angular * 0.28) / 0.16, (linear + angular * 0.28) / 0.16]
        for joint, speed in zip(self.wheels, speeds):
            joint.set_velocity(ecm, [float(speed)])
        for i, actor in enumerate(self.actor_states):
            x, y = actor["position"]
            self.models[f"actor_{i}"].set_world_pose_cmd(ecm, Pose3d(x, y, 0.6, 0, 0, 0))

    def _post_update(self, info, ecm):
        pose = self.robot_link.world_pose(ecm)
        self.pose = [pose.x(), pose.y(), pose.rot().yaw()]
        self.robot_position = np.array([pose.x(), pose.y(), pose.z()])
        roll, pitch, yaw = pose.rot().roll(), pose.rot().pitch(), pose.rot().yaw()
        cr, sr, cp, sp, cy, sy = (
            math.cos(roll),
            math.sin(roll),
            math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw),
            math.sin(yaw),
        )
        self.robot_rotation = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        velocity = self.robot_link.world_linear_velocity(ecm)
        angular = self.robot_link.world_angular_velocity(ecm)
        self.velocity = [
            velocity.x() * math.cos(self.pose[2]) + velocity.y() * math.sin(self.pose[2]),
            angular.z(),
        ]
        self.physics_time = info.sim_time.total_seconds()

    def step(self, data):
        self.control = data["control"]
        self.actor_states = data["actors"]
        steps = round(self.config["dt"] / min(self.config["dt"] / 5, 0.01))
        self.fixture.server().run(True, steps, False)
        # Contact.cc publishes only nonempty contact events, and reads the topic
        # inside <contact> (not <sensor>). A brief transport drain follows the
        # blocking physics tick; rendering also returns the latest latched label.
        deadline = time.monotonic() + 0.03
        while time.monotonic() < deadline:
            time.sleep(0.002)
        if self.transport_errors:
            raise RuntimeError("Native contact callback failed: " + self.transport_errors[0])
        return {
            "pose": self.pose,
            "velocity": self.velocity,
            "physics_time": self.physics_time - self.origin,
            "collision": bool(self.collision),
            "contact_stamps": self.contact_stamps,
            "contact_counts": self.contact_counts,
        }

    def render(self, data):
        name = "onboard" if data["source"] == "onboard" else "infrastructure"
        deadline = time.monotonic() + 10.0
        pair = None
        while time.monotonic() < deadline:
            with self.lock:
                rgb = self.frames.get((name, "rgb"))
                depth = self.frames.get((name, "depth"))
                calibration = self.calibrations.get(name)
                if (
                    rgb
                    and depth
                    and calibration is not None
                    and abs(rgb[0] - depth[0]) < 1e-5
                    and rgb[0] >= self.physics_time - 1e-5
                    and rgb[0] <= self.physics_time + 1e-5
                ):
                    pair = (rgb, depth)
            if pair is not None:
                break
            time.sleep(0.01)
        if pair is None:
            raise RuntimeError("Native Gazebo RGB-D did not publish; inspect renderer/plugin log")
        if self.transport_errors:
            raise RuntimeError("Native contact callback failed: " + self.transport_errors[0])
        rgb, depth = pair
        pose = data["pose"]
        rotation = position = None
        if name == "onboard":
            rotation = self.robot_rotation
            position = self.robot_position + rotation @ np.array([0.12, 0.0, 0.56])
            pose = [float(position[0]), float(position[1]), pose[2]]
        xy = camera_xy_map(
            depth[1],
            pose,
            0.85 if name == "onboard" else 3.0,
            0.0 if name == "onboard" else 0.6,
            self.config["camera_fov"],
            calibration,
            rotation,
            position,
        )
        self.counter += 1
        path = self.cache / f"frame_{self.counter}.npz"
        np.savez(path, rgb=rgb[1], depth=depth[1], xy_map=xy)
        return {
            "path": str(path),
            "capture_time": max(0.0, rgb[0] - self.origin),
            "pose": pose,
            "collision": bool(self.collision),
            "intrinsics": calibration.tolist(),
        }

    def close(self):
        # Harmonic Python Server.run(blocking=True) is already stopped on return;
        # the Python binding exposes no stop(). Release subscriptions and fixture.
        if self.node is not None:
            for topic in (
                "/robot_contacts",
                "/payload_contacts",
                "/onboard/image",
                "/onboard/depth_image",
                "/onboard/camera_info",
                "/infrastructure/image",
                "/infrastructure/depth_image",
                "/infrastructure/camera_info",
            ):
                self.node.unsubscribe(topic)
        self.node = None
        self.fixture = None


if __name__ == "__main__":
    serve(GazeboEngine)
