"""Offline SDF task generation. No remote assets or privileged perception labels."""

from __future__ import annotations
import math
import xml.etree.ElementTree as ET


def _node(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        node.text = str(text)
    return node


def _shape(link, name, geometry, color, friction=0.7):
    for tag in ("collision", "visual"):
        shape = _node(link, tag, name=name)
        geom = _node(shape, "geometry")
        kind = _node(geom, geometry[0])
        for key, value in geometry[1].items():
            _node(kind, key, value)
        if tag == "visual":
            material = _node(shape, "material")
            _node(material, "ambient", color)
            _node(material, "diffuse", color)
        else:
            surface = _node(shape, "surface")
            ode = _node(_node(surface, "friction"), "ode")
            _node(ode, "mu", friction)
            _node(ode, "mu2", friction)


def _link(model, name, pose, mass):
    link = _node(model, "link", name=name)
    _node(link, "pose", pose)
    inertia = _node(link, "inertial")
    _node(inertia, "mass", mass)
    tensor = _node(inertia, "inertia")
    # Conservative positive definite box approximation, overwritten for wheels.
    for key in ("ixx", "iyy", "izz"):
        _node(tensor, key, max(mass * 0.02, 0.0001))
    return link


def _camera(link, name, pose, config):
    sensor = _node(link, "sensor", name=name, type="rgbd_camera")
    _node(sensor, "pose", pose)
    _node(sensor, "topic", f"/{name}")
    _node(sensor, "always_on", "true")
    _node(sensor, "update_rate", 1.0 / config["dt"])
    cam = _node(sensor, "camera")
    _node(cam, "horizontal_fov", config["camera_fov"])
    image = _node(cam, "image")
    _node(image, "width", config["camera_width"])
    _node(image, "height", config["camera_height"])
    _node(image, "format", "R8G8B8")
    clip = _node(cam, "clip")
    _node(clip, "near", 0.05)
    _node(clip, "far", config["sensor_range"])


def _contact_sensor(link, name, collision, topic):
    sensor = _node(link, "sensor", name=name, type="contact")
    _node(sensor, "always_on", "true")
    _node(sensor, "topic", topic)
    contact = _node(sensor, "contact")
    _node(contact, "collision", collision)
    # Harmonic Contact.cc reads the topic inside <contact>.
    _node(contact, "topic", topic)


def gazebo_world(geometry, actors, config):
    """Generate one of the three corridor/junction task layouts for native transfer."""
    sdf = ET.Element("sdf", version="1.9")
    world = _node(sdf, "world", name="timelyhelp")
    _node(world, "gravity", "0 0 -9.81")
    physics = _node(world, "physics", name="default", type="ignored")
    _node(physics, "max_step_size", min(config["dt"] / 5, 0.01))
    _node(physics, "real_time_factor", 0)
    for filename, name in (
        ("physics", "Physics"),
        ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"),
        ("sensors", "Sensors"),
        ("contact", "Contact"),
    ):
        plugin = _node(
            world, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}"
        )
        if name == "Sensors":
            _node(plugin, "render_engine", "ogre2")
            _node(plugin, "headless_rendering", "true")
    light = _node(world, "light", name="sun", type="directional")
    _node(light, "pose", "0 0 8 0 0 0")
    _node(light, "diffuse", "1 1 1 1")
    _node(light, "direction", "0.2 0.2 -1")
    ground = _node(world, "model", name="ground")
    _node(ground, "static", "true")
    _shape(
        _link(ground, "link", "6 0 -0.1 0 0 0", 1),
        "ground",
        ("box", {"size": "30 20 0.2"}),
        "0.65 0.65 0.65 1",
    )
    for i, (x0, y0, x1, y1) in enumerate(geometry["rectangles"]):
        model = _node(world, "model", name=f"stack_{i}")
        _node(model, "static", "true")
        link = _link(model, "link", f"{(x0 + x1) / 2} {(y0 + y1) / 2} 0.8 0 0 0", 1)
        _shape(link, "box", ("box", {"size": f"{x1 - x0} {y1 - y0} 1.6"}), "0.4 0.4 0.4 1")
    start = geometry["start"]
    robot = _node(world, "model", name="robot")
    _node(robot, "pose", f"{start[0]} {start[1]} 0.16 0 0 {start[2]}")
    chassis = _link(robot, "chassis", "0 0 0.13 0 0 0", 18)
    _shape(chassis, "body", ("box", {"size": "0.6 0.48 0.2"}), "0.2 0.5 0.2 1")
    _contact_sensor(chassis, "chassis_contact", "body", "/robot_contacts")
    _camera(chassis, "onboard", "0.12 0 0.56 0 0 0", config)
    payload = _link(robot, "payload", "0 0 0.36 0 0 0", 12)
    _shape(payload, "load", ("box", {"size": "0.4 0.36 0.24"}), "0.6 0.5 0.2 1")
    _contact_sensor(payload, "payload_contact", "load", "/payload_contacts")
    joint = _node(robot, "joint", name="payload_fixed", type="fixed")
    _node(joint, "parent", "chassis")
    _node(joint, "child", "payload")
    for side, y in (("left", 0.28), ("right", -0.28)):
        link = _link(robot, f"{side}_wheel", f"0 {y} 0 1.57079632679 0 0", 0.8)
        _shape(
            link,
            "wheel",
            ("cylinder", {"radius": 0.16, "length": 0.07}),
            "0.1 0.1 0.1 1",
            config.get("wheel_friction", 0.7),
        )
        _contact_sensor(link, f"{side}_wheel_contact", "wheel", "/robot_contacts")
        joint = _node(robot, "joint", name=f"{side}_joint", type="revolute")
        _node(joint, "parent", "chassis")
        _node(joint, "child", f"{side}_wheel")
        axis = _node(joint, "axis")
        _node(axis, "xyz", "0 1 0", expressed_in="__model__")
        limit = _node(axis, "limit")
        _node(limit, "effort", 30)
        _node(limit, "velocity", 15)
    for x in (-0.23, 0.23):
        name = "rear_caster" if x < 0 else "front_caster"
        link = _link(robot, name, f"{x} 0 -0.08 0 0 0", 0.1)
        _shape(link, "caster", ("sphere", {"radius": 0.08}), "0.1 0.1 0.1 1", 0.01)
        _contact_sensor(link, f"{name}_contact", "caster", "/robot_contacts")
        joint = _node(robot, "joint", name=f"{name}_fixed", type="fixed")
        _node(joint, "parent", "chassis")
        _node(joint, "child", name)
    for i, actor in enumerate(actors):
        model = _node(world, "model", name=f"actor_{i}")
        x, y = actor["position"]
        _node(model, "pose", f"{x} {y} 0.6 0 0 0")
        link = _link(model, "body", "0 0 0 0 0 0", 60)
        _node(link, "gravity", "false")
        color = "0.9 0.16 0.06 1" if actor["kind"] == "worker" else "0.04 0.16 0.95 1"
        _shape(link, "actor", ("cylinder", {"radius": actor["radius"], "length": 1.2}), color)
    pose = geometry["camera_pose"]
    infrastructure = _node(world, "model", name="infrastructure")
    _node(infrastructure, "static", "true")
    link = _link(infrastructure, "camera_link", "0 0 0 0 0 0", 1)
    _camera(link, "infrastructure", f"{pose[0]} {pose[1]} 3.0 0 0.6 {pose[2]}", config)
    return ET.tostring(sdf, encoding="unicode")


def camera_xy_map(
    depth, pose, height, pitch, fov, intrinsics=None, camera_rotation=None, camera_position=None
):
    """Backproject metric optical-Z depth using only calibrated camera parameters."""
    import numpy as np

    rows, cols = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    # Empty native rays may contain infinity or NaN. Avoid undefined products
    # during projection, then mark their world coordinates unavailable.
    ray_depth = np.where(valid, depth, 0.0)
    if intrinsics is None:
        fx = fy = cols / (2 * math.tan(fov / 2))
        cx, cy = cols / 2, rows / 2
    else:
        calibration = np.asarray(intrinsics, dtype=float)
        if calibration.shape != (3, 3) or not np.isfinite(calibration).all():
            raise ValueError("Camera intrinsics must be a finite 3 by 3 matrix")
        fx, fy, cx, cy = calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
        if min(fx, fy) <= 0 or abs(calibration[0, 1]) > 1e-9:
            raise ValueError("Backprojection requires positive focal lengths and zero skew")
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    forward = ray_depth
    left = -(u - cx) / fx * ray_depth
    up = -(v - cy) / fy * ray_depth
    if camera_rotation is None:
        horizontal = math.cos(pitch) * forward + math.sin(pitch) * up
        x = pose[0] + math.cos(pose[2]) * horizontal - math.sin(pose[2]) * left
        y = pose[1] + math.sin(pose[2]) * horizontal + math.cos(pose[2]) * left
        xy = np.stack((x, y), axis=-1)
    else:
        rotation = np.asarray(camera_rotation, dtype=float)
        position = np.asarray(
            camera_position if camera_position is not None else [pose[0], pose[1], height],
            dtype=float,
        )
        if (
            rotation.shape != (3, 3)
            or position.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(position).all()
        ):
            raise ValueError("Camera extrinsics require a finite 3 by 3 rotation and 3D position")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-7
        ):
            raise ValueError("Camera rotation must be orthonormal and right handed")
        points = np.stack((forward, left, up), axis=-1) @ rotation.T + position
        xy = points[..., :2]
    xy[~valid] = np.nan
    return xy.astype(np.float32)
