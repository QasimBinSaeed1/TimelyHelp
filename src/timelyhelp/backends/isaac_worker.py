"""Isaac Sim 5.0: PhysX wheeled transport robot and native rendered cameras."""

from __future__ import annotations
import os
import math
import numpy as np
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "active_gpu": int(os.getenv("TIMELYHELP_GPU", "2")),
        "physics_gpu": int(os.getenv("TIMELYHELP_GPU", "2")),
        "multi_gpu": False,
        "width": 320,
        "height": 240,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 0,
        "disable_viewport_updates": True,
        "extra_args": [
            "--/app/hydraEngine/waitIdle=true",
            "--/app/updateOrder/checkForHydraRenderComplete=1000",
            "--/app/asyncRendering=false",
            "--/persistent/rtx/modes/rt/enabled=true",
            "--/persistent/rtx/modes/rt2/enabled=false",
            "--/persistent/rtx/modes/pt/enabled=false",
        ],
    }
)

import carb

render_settings = carb.settings.get_settings()
RENDER_SETTINGS = {
    "post/aa/op": 0,
    "post/tonemap/dither": 0.0,
    "sceneDb/ambientLightIntensity": 1.0,
    "sceneDb/ambientLightColor": [1.0, 1.0, 1.0],
    "raytracing/subpixel/filterRadius": 0.0,
    "raytracing/subpixel/uniformSubpixelJittering": False,
    "raytracing/subpixel/modulateRndSequence": False,
}
for key in (
    "post/tonemap/autoExposure/enabled",
    "ambientOcclusion/enabled",
    "indirectDiffuse/enabled",
    "reflections/enabled",
    "directLighting/enabled",
    "shadows/enabled",
    "post/motionblur/enabled",
    "post/dof/enabled",
):
    RENDER_SETTINGS[key] = False


def configure_rendering():
    # Use the SDK's zero-delay sensor profile. Copying a host buffer does not
    # wait for Hydra to finish writing it. Incomplete RGB rows can otherwise
    # change the observation-only controller across identical W/Q replays.
    render_settings.set("/app/hydraEngine/waitIdle", True)
    render_settings.set("/app/updateOrder/checkForHydraRenderComplete", 1000)
    render_settings.set("/app/asyncRendering", False)
    render_settings.set("/rtx/materialDb/syncLoads", True)
    render_settings.set("/rtx/hydra/materialSyncLoads", True)
    render_settings.set("/rtx/ecoMode/enabled", False)
    # Render products restore the default profile on every render. Set both
    # profiles, rather than changing only the transient active viewport values.
    # Kit's default limitedOps permits only DLSS/DLAA and silently coerces "none"
    # back to DLSS, which jitters primary rays even below DLSS's minimum resolution.
    render_settings.set("/rtx-transient/post/aa/limitedOps", False)
    for prefix in ("/rtx", "/rtx-defaults"):
        for key, value in RENDER_SETTINGS.items():
            render_settings.set(f"{prefix}/{key}", value)


configure_rendering()

from pxr import (
    Gf,
    Sdf,
    UsdGeom,
    UsdPhysics,
    PhysxSchema,
    UsdLux,
    PhysicsSchemaTools,
    UsdShade,
    Usd,
)
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.materials import PreviewSurface
from worlds import camera_xy_map
from worker_common import serve


class IsaacEngine:
    def __init__(self, cache):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.world = None
        self.counter = 0
        self.collision = False
        from omni.physx import get_physx_simulation_interface

        self.contact_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(self._contacts)
        )

    def _contacts(self, headers, contact_data):
        for header in headers:
            paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            if any(path.startswith("/World/Robot") for path in paths):
                foreign = [path for path in paths if not path.startswith("/World/Robot")]
                if any(
                    "groundPlane" not in path and "defaultGroundPlane" not in path
                    for path in foreign
                ):
                    # CONTACT_LOST does not start a collision; nonempty data confirms touch.
                    if header.num_contact_data > 0:
                        self.collision = True

    def _visual(self, prim, name, color):
        """Constant-radiance class markers rendered by the actual RTX camera.

        Frozen markers isolate scheduling from photometric Monte Carlo noise;
        native occlusion, projection and RGB-D acquisition remain unchanged.
        """
        material = PreviewSurface(
            prim_path=f"/World/Visual_{name}", color=np.zeros(3), roughness=1.0, metallic=0.0
        )
        material.shaders_list[0].CreateInput("emissiveColor", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(*color)
        )
        material.shaders_list[0].CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
        material.shaders_list[0].CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    def _configure_products(self, stage, presentation=False):
        # Kit 107 uses authoritative USD settings on each render product. Its
        # compatibility layer overwrites deprecated /rtx settings every frame.
        settings = {
            "omni:rtx:rendermode": (Sdf.ValueTypeNames.Token, "RaytracedLighting"),
            "omni:rtx:post:aa:op": (Sdf.ValueTypeNames.Token, "none"),
            "omni:rtx:post:aa:limitedOps": (Sdf.ValueTypeNames.Bool, False),
            "omni:rtx:post:tonemap:dither": (Sdf.ValueTypeNames.Float, 0.0),
            "omni:rtx:rt:ambientLight:color": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1, 1, 1)),
            "omni:rtx:rt:ambientLight:intensity": (Sdf.ValueTypeNames.Float, 1.0),
            "omni:rtx:rt:subpixel:filterRadius": (Sdf.ValueTypeNames.Float, 0.0),
            "omni:rtx:rt:subpixel:uniformSubpixelJittering": (Sdf.ValueTypeNames.Bool, False),
            "omni:rtx:rt:subpixel:modulateRndSequence": (Sdf.ValueTypeNames.Bool, False),
        }
        for key in (
            "rt:directLighting:enabled",
            "rt:indirectdiffuse:enabled",
            "rt:ambientocclusion:enabled",
            "rt:reflections:enabled",
            "shadows:enabled",
            "autoExposure:enabled",
            "post:motionblur:enabled",
            "post:dof:enabled",
        ):
            settings[f"omni:rtx:{key}"] = (Sdf.ValueTypeNames.Bool, False)
        if presentation:
            settings.update(
                {
                    "omni:rtx:rt:ambientLight:intensity": (Sdf.ValueTypeNames.Float, 0.25),
                    "omni:rtx:rt:directLighting:enabled": (Sdf.ValueTypeNames.Bool, True),
                    "omni:rtx:rt:ambientocclusion:enabled": (Sdf.ValueTypeNames.Bool, True),
                    "omni:rtx:shadows:enabled": (Sdf.ValueTypeNames.Bool, True),
                }
            )
        import omni.usd
        import usdrt

        fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        paths = {camera.get_render_product_path() for camera in self.cameras.values()}
        paths.update(camera.prim_path for camera in self.cameras.values())
        paths.update(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetTypeName() == "RenderProduct"
        )
        paths.update(str(path) for path in fabric_stage.GetPrimsWithTypeName("RenderProduct"))
        if presentation:
            paths = {
                self.cameras["viewer"].get_render_product_path(),
                self.cameras["viewer"].prim_path,
            }
        for path in sorted(paths):
            prim = stage.GetPrimAtPath(path)
            fabric_prim = fabric_stage.GetPrimAtPath(path)
            if not prim.IsValid() or not fabric_prim.IsValid():
                raise RuntimeError(f"Camera render product unavailable: {path}")
            for key, (kind, value) in settings.items():
                # Replicator products live in the stronger session layer.
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    prim.CreateAttribute(key, kind, custom=False).Set(value)
                # Fabric render profiles can already contain authored defaults;
                # set their corresponding attributes too, rather than relying
                # on USD compatibility synchronization to overwrite them.
                attribute = fabric_prim.GetAttribute(key)
                if not attribute:
                    kind_name = {
                        Sdf.ValueTypeNames.Token: "Token",
                        Sdf.ValueTypeNames.Float: "Float",
                        Sdf.ValueTypeNames.Bool: "Bool",
                        Sdf.ValueTypeNames.Color3f: "Color3f",
                    }[kind]
                    attribute = fabric_prim.CreateAttribute(
                        key, getattr(usdrt.Sdf.ValueTypeNames, kind_name), False
                    )
                if kind == Sdf.ValueTypeNames.Color3f:
                    value = usdrt.Gf.Vec3f(*value)
                attribute.Set(value)

    def reset(self, data):
        import omni.usd

        if self.world is not None:
            self.world.stop()
            self.world.clear_instance()
            # Camera render products and their shared SyntheticData graph must
            # survive resets. Isaac 5.0 annotator teardown can leave invalid
            # graph nodes when a second camera shares the pipeline. Rebuild the
            # physical world on the same stage, retaining the camera prims and
            # render products; no solver state survives World.reset().
            stage = omni.usd.get_context().get_stage()
            for prim in list(stage.GetPrimAtPath("/World").GetChildren()):
                if not prim.GetName().startswith("Camera_"):
                    stage.RemovePrim(prim.GetPath())
        else:
            omni.usd.get_context().new_stage()
            self.cameras = {}
        configure_rendering()
        self.config, self.geometry = data["config"], data["geometry"]
        self.presentation = None
        self.presentation_actors = data["actors"]
        self.presentation_setting = data.get("presentation_setting", "worksite")
        self.world = World(stage_units_in_meters=1, physics_dt=0.01, rendering_dt=self.config["dt"])
        stage = omni.usd.get_context().get_stage()
        # DynaNav reference assets can be used as background without changing its
        # original benchmark scripts or claiming its pretrained maps are unseen.
        scene = os.getenv("TIMELYHELP_DYNANAV_SCENE")
        if scene:
            prim = UsdGeom.Xform.Define(stage, "/World/DynaNavScene")
            prim.GetPrim().GetReferences().AddReference(scene)
        self.world.scene.add_default_ground_plane()
        self._visual(stage.GetPrimAtPath("/World/defaultGroundPlane"), "Ground", [0.65, 0.65, 0.65])
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        # Fixed ambient illumination keeps the marker encoder and A/W/Q sensors
        # deterministic. Stochastic RTX light sampling is a different nuisance
        # draw unless its complete render state can also be restored.
        light.CreateIntensityAttr(0)
        for i, (x0, y0, x1, y1) in enumerate(self.geometry["rectangles"]):
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Stack_{i}",
                    name=f"stack_{i}",
                    position=np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.8]),
                    scale=np.array([x1 - x0, y1 - y0, 1.6]),
                    color=np.array([0.4, 0.4, 0.4]),
                )
            )
            self._visual(stage.GetPrimAtPath(f"/World/Stack_{i}"), f"Stack_{i}", [0.4, 0.4, 0.4])
        root = UsdGeom.Xform.Define(stage, "/World/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateEnabledSelfCollisionsAttr(
            False
        )
        sx, sy, yaw = self.geometry["start"]
        base = "/World/Robot/Chassis"
        self.world.scene.add(
            DynamicCuboid(
                prim_path=base,
                name="chassis",
                mass=30,
                position=np.array([sx, sy, 0.29]),
                scale=np.array([0.6, 0.48, 0.2]),
                color=np.array([0.2, 0.5, 0.2]),
            )
        )
        self._visual(stage.GetPrimAtPath(base), "Chassis", [0.2, 0.5, 0.2])
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(base)).CreateThresholdAttr(0)
        load = UsdGeom.Cube.Define(stage, base + "/Payload")
        load.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.5))
        load.AddScaleOp().Set(Gf.Vec3d(0.65, 0.7, 1.2))
        # Payload is part of the chassis rigid body: fixed transport load, no arm.
        UsdPhysics.CollisionAPI.Apply(load.GetPrim())
        wheel_material = UsdShade.Material.Define(stage, "/World/WheelPhysicsMaterial")
        wheel_physics = UsdPhysics.MaterialAPI.Apply(wheel_material.GetPrim())
        wheel_physics.CreateStaticFrictionAttr(self.config.get("wheel_friction", 0.7))
        wheel_physics.CreateDynamicFrictionAttr(self.config.get("wheel_friction", 0.7))
        for side, y in (("left", 0.28), ("right", -0.28)):
            path = f"/World/Robot/{side}_wheel"
            cylinder = UsdGeom.Cylinder.Define(stage, path)
            cylinder.CreateRadiusAttr(0.16)
            cylinder.CreateHeightAttr(0.07)
            cylinder.CreateAxisAttr("Y")
            cylinder.AddTranslateOp().Set(Gf.Vec3d(sx, sy + y, 0.16))
            cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])
            self._visual(cylinder.GetPrim(), f"Wheel_{side}", [0.08, 0.08, 0.08])
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
            PhysxSchema.PhysxContactReportAPI.Apply(cylinder.GetPrim()).CreateThresholdAttr(0)
            UsdPhysics.MassAPI.Apply(cylinder.GetPrim()).CreateMassAttr(0.8)
            UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(
                wheel_material, materialPurpose="physics"
            )
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{side}_joint")
            joint.CreateBody0Rel().SetTargets([base])
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, y, -0.13))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateDampingAttr(400)
            drive.CreateStiffnessAttr(0)
            drive.CreateMaxForceAttr(40)
        for i, x in enumerate((-0.23, 0.23)):
            path = f"/World/Robot/Caster_{i}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(0.08)
            sphere.AddTranslateOp().Set(Gf.Vec3d(sx + x, sy, 0.08))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
            UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.1)
            joint = UsdPhysics.FixedJoint.Define(stage, f"/World/Robot/CasterJoint_{i}")
            joint.CreateBody0Rel().SetTargets([base])
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos0Attr(Gf.Vec3f(x, 0, -0.21))
            # Frictionless support casters avoid constraining differential steering.
            caster_material = UsdShade.Material.Define(stage, path + "/Material")
            material = UsdPhysics.MaterialAPI.Apply(caster_material.GetPrim())
            material.CreateStaticFrictionAttr(0.01)
            material.CreateDynamicFrictionAttr(0.01)
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                caster_material, materialPurpose="physics"
            )
        self.robot = self.world.scene.add(
            SingleArticulation(prim_path="/World/Robot", name="robot")
        )
        self.actor_prims = []
        for i, actor in enumerate(data["actors"]):
            cylinder = UsdGeom.Cylinder.Define(stage, f"/World/Actor_{i}")
            cylinder.CreateRadiusAttr(actor["radius"])
            cylinder.CreateHeightAttr(1.2)
            cylinder.CreateAxisAttr("Z")
            transform = cylinder.AddTranslateOp()
            x, y = actor["position"]
            transform.Set(Gf.Vec3d(x, y, 0.6))
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            body = UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
            body.CreateKinematicEnabledAttr(True)
            color = (
                np.array([0.9, 0.16, 0.06])
                if actor["kind"] == "worker"
                else np.array([0.04, 0.16, 0.95])
            )
            cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            self._visual(cylinder.GetPrim(), f"Actor_{i}", color)
            self.actor_prims.append(transform)
        for name in ("onboard", "infrastructure"):
            if name not in self.cameras:
                camera = Camera(
                    prim_path=f"/World/Camera_{name}",
                    frequency=round(1 / self.config["dt"]),
                    resolution=(self.config["camera_width"], self.config["camera_height"]),
                )
                self.cameras[name] = camera
        self.world.reset()
        self.wheel_indices = [
            self.robot.get_dof_index(f"{side}_joint") for side in ("left", "right")
        ]
        for name, camera in self.cameras.items():
            if name == "viewer":
                # A human-only diagnostic camera retains its independent calibration.
                continue
            if not camera._custom_annotators:
                camera.initialize()
                camera.add_distance_to_image_plane_to_frame()
            camera.set_frequency(round(1 / self.config["dt"]))
            camera.set_resolution((self.config["camera_width"], self.config["camera_height"]))
            # Frequency gating on get_current_frame skips annotator updates when
            # several renders share one physics tick. Direct annotator reads
            # plus an uncopied buffer were enough to mix whole RGB rows.
            camera._frequency = -1
            camera._previous_time = None
            camera._elapsed_time = 0
            camera.set_focal_length(12)
            camera.set_horizontal_aperture(24 * math.tan(self.config["camera_fov"] / 2))
            camera.set_clipping_range(0.05, self.config["sensor_range"])
        self._configure_products(stage)
        self._position_cameras([sx, sy, yaw])
        for _ in range(50):
            self.world.step(render=False)
        # Settling needs physics, not fifty duplicated camera render passes.
        for _ in range(3):
            self.world.render()
        self.origin = self.world.current_time
        self.steps = 0
        self.collision = False
        return {"backend": "isaac", "pose": [sx, sy, yaw], "collision": False}

    def _position_cameras(self, pose):
        x, y, yaw = pose
        self.cameras["onboard"].set_world_pose(
            np.array([x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw), 0.85]),
            euler_angles_to_quats(np.array([0, 0, yaw])),
            camera_axes="world",
        )
        x, y, yaw = self.geometry["camera_pose"]
        self.cameras["infrastructure"].set_world_pose(
            np.array([x, y, 3.0]),
            euler_angles_to_quats(np.array([0, 0.6, yaw])),
            camera_axes="world",
        )

    def step(self, data):
        self.presentation_actors = data["actors"]
        linear, angular = data["control"]
        action = ArticulationAction(
            joint_velocities=np.array(
                [(linear - angular * 0.28) / 0.16, (linear + angular * 0.28) / 0.16]
            ),
            joint_indices=np.array(self.wheel_indices),
        )
        self.robot.apply_action(action)
        for transform, actor in zip(self.actor_prims, data["actors"]):
            x, y = actor["position"]
            transform.Set(Gf.Vec3d(x, y, 0.6))
        for _ in range(round(self.config["dt"] / 0.01)):
            self.world.step(render=False)
        position, quaternion = self.robot.get_world_pose()
        w, x, y, z = quaternion
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pose = [float(position[0]), float(position[1]), yaw]
        velocity = self.robot.get_linear_velocity()
        angular_velocity = self.robot.get_angular_velocity()
        self.steps += 1
        return {
            "pose": pose,
            "velocity": [
                float(velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw)),
                float(angular_velocity[2]),
            ],
            "physics_time": self.world.current_time - self.origin,
            "collision": bool(self.collision),
        }

    def _aim_camera(self, name, pose):
        """Set the requested product's calibrated camera pose before capture."""
        x, y, yaw = pose
        camera = self.cameras[name]
        if name == "onboard":
            camera.set_world_pose(
                np.array([x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw), 0.85]),
                euler_angles_to_quats(np.array([0, 0, yaw])),
                camera_axes="world",
            )
        else:
            camera.set_world_pose(
                np.array([x, y, 3.0]),
                euler_angles_to_quats(np.array([0, 0.6, yaw])),
                camera_axes="world",
            )
        return camera

    def _set_product_updates(self, camera, enabled):
        product = getattr(camera, "_render_product", None)
        texture = getattr(product, "hydra_texture", None) if product is not None else None
        if texture is not None and hasattr(texture, "set_updates_enabled"):
            texture.set_updates_enabled(bool(enabled))

    def _annotator_array(self, camera, name):
        annotators = getattr(camera, "_custom_annotators", {})
        annotator = annotators.get(name)
        data = None
        if annotator is not None:
            data = annotator.get_data(do_array_copy=True)
        if data is None and name == "rgb" and hasattr(camera, "get_rgba"):
            data = camera.get_rgba()
        if data is None and name == "distance_to_image_plane" and hasattr(camera, "get_depth"):
            data = camera.get_depth()
        if data is None:
            return None
        return np.array(data, copy=True)

    def _read_rgbd(self, camera):
        rgb = self._annotator_array(camera, "rgb")
        depth = self._annotator_array(camera, "distance_to_image_plane")
        if rgb is None or depth is None:
            frame = (
                camera.get_current_frame(clone=True) if hasattr(camera, "get_current_frame") else {}
            )
            if rgb is None and isinstance(frame, dict):
                rgb = frame.get("rgba")
                rgb = None if rgb is None else np.array(rgb, copy=True)
            if depth is None and isinstance(frame, dict):
                depth = frame.get("distance_to_image_plane")
                depth = None if depth is None else np.array(depth, copy=True)
        if rgb is None or depth is None:
            raise RuntimeError("Isaac camera produced no RGB-D frame")
        rgb = np.array(rgb[..., :3], copy=True)
        depth = np.array(depth, copy=True)
        if rgb.size == 0 or depth.size == 0:
            raise RuntimeError("Isaac camera produced no RGB-D frame")
        return rgb, depth

    def render(self, data):
        configure_rendering()
        import omni.usd

        self._configure_products(omni.usd.get_context().get_stage())
        name = "onboard" if data["source"] == "onboard" else "infrastructure"
        camera = self._aim_camera(name, data["pose"])
        # Both scientific products keep the same schedule in every branch. A
        # diagnostic camera must not change that schedule. Read RGB and depth
        # from one completed frame; a median of separately acquired colors can
        # create a color/depth pair that the camera never actually produced.
        for other, other_camera in self.cameras.items():
            self._set_product_updates(other_camera, other != "viewer")
        physics_time = self.world.current_time
        for _ in range(3):
            self.world.render()
        rgb, depth = self._read_rgbd(camera)
        if self.world.current_time != physics_time:
            raise RuntimeError("Sensor acquisition advanced physics time")
        expected = (self.config["camera_height"], self.config["camera_width"])
        if rgb.shape != (*expected, 3) or depth.shape != expected:
            raise RuntimeError(f"Invalid native RGB-D dimensions: {rgb.shape}, {depth.shape}")
        position, _ = camera.get_world_pose(camera_axes="world")
        pose = data["pose"]
        if name == "onboard":
            pose = [float(position[0]), float(position[1]), pose[2]]
        xy = camera_xy_map(
            depth,
            pose,
            position[2],
            0 if name == "onboard" else 0.6,
            self.config["camera_fov"],
            camera.get_intrinsics_matrix(),
        )
        self.counter += 1
        path = self.cache / f"frame_{self.counter}.npz"
        np.savez(path, rgb=rgb, depth=depth, xy_map=xy)
        return {
            "path": str(path),
            "pose": pose,
            "capture_time": self.world.current_time - self.origin,
            "render_product": camera.get_render_product_path(),
            "synchronized": True,
        }

    def viewer_frame(self, data):
        """Render an optional diagnostic overview; never expose it to a policy."""
        import omni.usd

        if "viewer" not in self.cameras:
            camera = Camera(
                prim_path="/World/Camera_viewer",
                frequency=round(1 / self.config["dt"]),
                resolution=(1024, 576),
            )
            camera.initialize()
            camera.set_focal_length(12)
            camera.set_horizontal_aperture(24)
            camera.set_clipping_range(0.05, 100)
            self.cameras["viewer"] = camera
        camera = self.cameras["viewer"]
        center = float(self.geometry["goal"][0]) / 2
        view = data.get("view", "overview")
        if view not in ("overview", "top", "follow", "orbit"):
            raise ValueError("Unknown diagnostic camera view")
        if view == "follow":
            position, _ = self.robot.get_world_pose()
            target = Gf.Vec3d(float(position[0]) + 0.7, float(position[1]), 0.65)
            eye = Gf.Vec3d(float(position[0]) - 2.5, float(position[1]) - 3.2, 2.7)
        elif view == "top":
            target = Gf.Vec3d(center, 0, 0)
            eye = Gf.Vec3d(center, -0.01, 13)
        elif view == "orbit":
            from presentation import orbit_camera

            target = Gf.Vec3d(center, 0, 0.6)
            eye = Gf.Vec3d(
                *orbit_camera(
                    target,
                    data.get("azimuth", -55),
                    data.get("elevation", 34),
                    data.get("distance", 12),
                )
            )
        else:
            target = Gf.Vec3d(center, 0.7, 0.6)
            eye = Gf.Vec3d(center - 7, -8, 6)
        transform = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        quaternion = transform.ExtractRotationQuat()
        orientation = np.array([quaternion.GetReal(), *quaternion.GetImaginary()])
        camera.set_world_pose(np.array(eye), orientation, camera_axes="usd")
        configure_rendering()
        stage = omni.usd.get_context().get_stage()
        self._configure_products(stage)
        from presentation import PresentationScene

        if self.presentation is None:
            self.presentation = PresentationScene(
                stage,
                self.geometry,
                self.presentation_actors,
                data.get("setting", self.presentation_setting),
            )
            self.presentation.validate_nonphysical()
        position, quaternion = self.robot.get_world_pose()
        w, x, y, z = quaternion
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.presentation.update(
            [position[0], position[1], yaw],
            self.presentation_actors,
            self.world.current_time - self.origin,
        )
        # Visibility is render-only. Physical bodies are never deactivated and no
        # physics step occurs here. Restore even if image acquisition fails.
        hidden = []
        try:
            for prim in stage.GetPrimAtPath("/World").GetChildren():
                if prim.GetName() in (
                    "Robot",
                    "defaultGroundPlane",
                    "DynaNavScene",
                ) or prim.GetName().startswith(("Actor_", "Stack_")):
                    attribute = UsdGeom.Imageable(prim).CreateVisibilityAttr()
                    hidden.append((attribute, attribute.Get()))
                    attribute.Set(UsdGeom.Tokens.invisible)
            self.presentation.visible(True)
            self._configure_products(stage, presentation=True)
            self._set_product_updates(camera, True)
            for _ in range(5):
                self.world.render()
            rgba = self._annotator_array(camera, "rgb")
            if rgba is None:
                rgba = np.array(camera.get_rgba(), copy=True)
        finally:
            self.presentation.visible(False)
            for attribute, value in hidden:
                attribute.Set(value)
            configure_rendering()
            self._configure_products(stage)
            self._set_product_updates(camera, False)
            # Refresh camera annotators in their original scientific scene so a
            # query immediately after this RPC cannot consume presentation pixels.
            for _ in range(3):
                self.world.render()
        if rgba is None or rgba.size == 0:
            raise RuntimeError("Isaac diagnostic camera produced no image")
        self.counter += 1
        path = self.cache / f"viewer_{self.counter}.npz"
        np.savez(path, rgb=rgba[..., :3])
        return {
            "path": str(path),
            "capture_time": self.world.current_time - self.origin,
            "presentation": "detailed_nonphysical",
            "sensor_scene_restored": True,
            "presentation_assets": self.presentation.asset_manifest,
        }

    def close(self):
        app.close()


if __name__ == "__main__":
    serve(IsaacEngine)
