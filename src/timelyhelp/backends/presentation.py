"""Detailed human inspection view; no physics, perception or policy inputs.

The diagnostic render temporarily shows this subtree and hides the marker meshes.
Every visibility/profile change is restored before research sensors render. Mesh
transforms follow measured robot pose and reactive actor positions, never a second
simulation. Decorative limbs follow a deterministic gait, not new actor dynamics.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path


def orbit_camera(center, azimuth=-55, elevation=34, distance=12):
    if not all(math.isfinite(float(v)) for v in (*center, azimuth, elevation, distance)):
        raise ValueError("Camera parameters must be finite")
    elevation = min(85, max(10, float(elevation)))
    distance = min(30, max(2, float(distance)))
    a, e = math.radians(float(azimuth)), math.radians(elevation)
    return [
        center[0] + distance * math.cos(e) * math.cos(a),
        center[1] + distance * math.cos(e) * math.sin(a),
        center[2] + distance * math.sin(e),
    ]


class PresentationScene:
    root_path = "/World/Presentation"

    def __init__(self, stage, geometry, actors, setting="worksite"):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        self.Gf, self.Sdf, self.UsdGeom, self.UsdShade = Gf, Sdf, UsdGeom, UsdShade
        self.stage = stage
        self.geometry = geometry
        self.root = UsdGeom.Xform.Define(stage, self.root_path)
        self.root.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        self.materials = {}
        self.models = []
        self.asset_manifest = {}
        self.setting = setting
        self._materials()
        self._environment()
        self.robot = self._robot()
        for i, actor in enumerate(actors):
            self.models.append(self._human(i) if actor["kind"] == "worker" else self._cart(i))
        self.update([*geometry["start"]], actors, 0)

    def _material(self, name, color, metallic=0, roughness=0.6, emission=0):
        material = self.UsdShade.Material.Define(self.stage, f"{self.root_path}/Materials/{name}")
        shader = self.UsdShade.Shader.Define(self.stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        for key, kind, value in (
            ("diffuseColor", self.Sdf.ValueTypeNames.Color3f, self.Gf.Vec3f(*color)),
            ("metallic", self.Sdf.ValueTypeNames.Float, metallic),
            ("roughness", self.Sdf.ValueTypeNames.Float, roughness),
            (
                "emissiveColor",
                self.Sdf.ValueTypeNames.Color3f,
                self.Gf.Vec3f(*(c * emission for c in color)),
            ),
        ):
            shader.CreateInput(key, kind).Set(value)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.materials[name] = material

    def _materials(self):
        palette = {
            "floor": (0.20, 0.24, 0.28),
            "panel": (0.09, 0.13, 0.19),
            "wall": (0.35, 0.43, 0.52),
            "yellow": (0.97, 0.59, 0.03),
            "white": (0.82, 0.87, 0.91),
            "black": (0.012, 0.018, 0.026),
            "steel": (0.30, 0.37, 0.45),
            "blue": (0.02, 0.20, 0.45),
            "wood": (0.38, 0.20, 0.08),
            "crate": (0.55, 0.34, 0.15),
            "green": (0.07, 0.45, 0.28),
            "red": (0.75, 0.09, 0.035),
            "orange": (1.0, 0.25, 0.015),
            "skin": (0.58, 0.32, 0.20),
            "pavement": (0.44, 0.46, 0.47),
            "asphalt": (0.075, 0.085, 0.10),
        }
        for name, color in palette.items():
            self._material(
                name,
                color,
                metallic=0.65 if name == "steel" else 0,
                roughness=0.3 if name == "steel" else 0.68,
                emission=0.015,
            )
        self._material("light", (0.8, 0.9, 1), emission=6)
        self._material("cyan", (0.05, 0.75, 1), emission=2)

    def _textured_material(self, name, texture_path):
        """Portable PreviewSurface equivalent of the asset's MDL base-color map."""
        if not texture_path.is_file():
            raise RuntimeError(f"DynaNav human texture is missing: {texture_path}")
        self.asset_manifest[str(texture_path)] = hashlib.sha256(
            texture_path.read_bytes()
        ).hexdigest()
        self._material(name, (1, 1, 1), roughness=0.8)
        material = self.materials[name]
        path = str(material.GetPath())
        reader = self.UsdShade.Shader.Define(self.stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", self.Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", self.Sdf.ValueTypeNames.Float2)
        texture = self.UsdShade.Shader.Define(self.stage, path + "/BaseColor")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", self.Sdf.ValueTypeNames.Asset).Set(
            self.Sdf.AssetPath(str(texture_path))
        )
        texture.CreateInput("sourceColorSpace", self.Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", self.Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        texture.CreateOutput("rgb", self.Sdf.ValueTypeNames.Float3)
        shader = self.UsdShade.Shader(self.stage.GetPrimAtPath(path + "/Surface"))
        shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")

    def _bind(self, prim, material):
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.materials[material])

    def _box(self, path, position, size, material):
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(self.Gf.Vec3d(*position))
        cube.AddScaleOp().Set(self.Gf.Vec3d(*size))
        self._bind(cube.GetPrim(), material)
        return cube

    def _cylinder(self, path, position, radius, height, material, axis="Z"):
        cylinder = self.UsdGeom.Cylinder.Define(self.stage, path)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        cylinder.CreateAxisAttr(axis)
        cylinder.AddTranslateOp().Set(self.Gf.Vec3d(*position))
        self._bind(cylinder.GetPrim(), material)
        return cylinder

    def _group(self, path, position=(0, 0, 0)):
        group = self.UsdGeom.Xform.Define(self.stage, path)
        translation = group.AddTranslateOp()
        translation.Set(self.Gf.Vec3d(*position))
        rotation = group.AddRotateZOp()
        return group, translation, rotation

    def _pallet(self, path, x, y, z=0, scale=1):
        self._group(path, (x, y, z))
        for i in range(5):
            self._box(
                f"{path}/Plank_{i}",
                (0, (i - 2) * 0.20 * scale, 0.13),
                (1.1 * scale, 0.15 * scale, 0.05),
                "wood",
            )
        for i in range(3):
            self._box(
                f"{path}/Runner_{i}",
                ((i - 1) * 0.42 * scale, 0, 0.065),
                (0.10 * scale, 0.98 * scale, 0.10),
                "wood",
            )
        self._box(path + "/Load", (0, 0, 0.58), (0.92 * scale, 0.87 * scale, 0.84), "crate")
        for i in (-1, 1):
            self._box(
                f"{path}/Band_{i + 1}",
                (i * 0.28 * scale, 0, 1.005),
                (0.04, 0.88 * scale, 0.01),
                "black",
            )

    def _environment(self):
        from pxr import UsdLux

        r = self.root_path + "/Environment"
        goal = self.geometry["goal"][0]
        self._box(r + "/Slab", (goal / 2, 0, -0.08), (goal + 8, 16, 0.16), "floor")
        # Scene geometry shows the same obstacle envelopes as the collision model.
        for i, (x0, y0, x1, y1) in enumerate(self.geometry["rectangles"]):
            x, y = (x0 + x1) / 2, (y0 + y1) / 2
            if i < 2:
                self._box(f"{r}/Barrier_{i}", (x, y, 0.32), (x1 - x0, y1 - y0, 0.64), "steel")
                for j in range(round(x1 - x0) + 1):
                    self._box(f"{r}/Post_{i}_{j}", (x0 + j, y, 0.92), (0.075, 0.075, 1.2), "yellow")
                self._box(f"{r}/Rail_{i}", (x, y, 1.50), (x1 - x0, 0.065, 0.07), "yellow")
            else:
                self._box(f"{r}/Occluder_{i}", (x, y, 0.8), (x1 - x0, y1 - y0, 1.6), "crate")
                for z in (0.15, 0.8, 1.45):
                    self._box(
                        f"{r}/CrateBand_{i}_{int(z * 100)}",
                        (x, y - (y1 - y0) / 2 - 0.008, z),
                        (x1 - x0, 0.02, 0.045),
                        "black",
                    )
        for side in (-1, 1):
            self._box(
                f"{r}/Lane_{side + 1}",
                (goal / 2, side * 1.9, 0.006),
                (goal + 0.3, 0.055, 0.012),
                "yellow",
            )
        crossing = self.geometry["crossing"]
        for i in range(8):
            self._box(
                f"{r}/Crosswalk_{i}",
                (crossing, (i - 3.5) * 0.48, 0.008),
                (1.4, 0.24, 0.015),
                "white",
            )
        self._box(r + "/GoalPad", (goal, 0, 0.009), (1.1, 1.4, 0.016), "green")
        self._box(r + "/Dock", (goal + 0.65, 0, 0.14), (0.45, 1.8, 0.28), "steel")
        if self.setting == "sidewalk":
            self._box(r + "/Sidewalk", (goal / 2, 0, 0.002), (goal + 3, 5.2, 0.01), "pavement")
            self._box(r + "/Road", (goal / 2, -5, -0.02), (goal + 8, 3.8, 0.025), "asphalt")
            for i in range(6):
                self._box(f"{r}/RoadDash_{i}", (i * 2.6, -5, 0.001), (1.1, 0.09, 0.015), "yellow")
                self._box(f"{r}/Building_{i}", (i * 2.6, 5.4, 2.1), (2.4, 2.5, 4.2), "panel")
                for j in range(2):
                    self._box(
                        f"{r}/Window_{i}_{j}",
                        (i * 2.6, 4.13, 1.5 + j * 1.5),
                        (1.55, 0.025, 0.85),
                        "cyan",
                    )
                self._cylinder(f"{r}/TreeTrunk_{i}", (i * 2.6, 3.5, 0.9), 0.10, 1.8, "wood")
                sphere = self.UsdGeom.Sphere.Define(self.stage, f"{r}/TreeCrown_{i}")
                sphere.CreateRadiusAttr(0.65)
                sphere.AddTranslateOp().Set(self.Gf.Vec3d(i * 2.6, 3.5, 2.2))
                self._bind(sphere.GetPrim(), "green")
        else:
            self._box(r + "/BackWall", (goal / 2, 7, 2.7), (goal + 6, 0.2, 5.4), "panel")
            for side, x in enumerate((-3, goal + 3)):
                self._box(f"{r}/EndWall_{side}", (x, 0, 2.7), (0.2, 14, 5.4), "panel")
                self._box(
                    f"{r}/LoadingDoor_{side}",
                    (x + (0.11 if side == 0 else -0.11), 0, 1.65),
                    (0.025, 3.0, 3.3),
                    "steel",
                )
                for strip in range(12):
                    self._box(
                        f"{r}/DoorRib_{side}_{strip}",
                        (x + (0.13 if side == 0 else -0.13), 0, 0.15 + strip * 0.27),
                        (0.015, 3.0, 0.025),
                        "black",
                    )
            for i in range(4):
                x = i * 3.2 + 0.4
                rack = f"{r}/Rack_{i}"
                for dx in (-1.15, 1.15):
                    for dy in (-0.55, 0.55):
                        self._box(
                            f"{rack}/Post_{int((dx + 2) * 100)}_{int((dy + 1) * 100)}",
                            (x + dx, 4.8 + dy, 1.8),
                            (0.085, 0.085, 3.6),
                            "blue",
                        )
                for z in (0.15, 1.5, 2.85):
                    self._box(
                        f"{rack}/Shelf_{int(z * 100)}", (x, 4.8, z), (2.5, 1.25, 0.10), "steel"
                    )
                    self._pallet(f"{rack}/Pallet_{int(z * 100)}", x, 4.8, z + 0.04, 0.85)
                self._box(f"{r}/Window_{i}", (x, 6.88, 4.1), (2.3, 0.025, 0.6), "cyan")
                self._box(f"{r}/CeilingBeam_{i}", (x, 0, 4.8), (0.15, 13, 0.18), "steel")
                self._box(f"{r}/LED_{i}", (x, 1.6, 4.68), (0.08, 2.6, 0.035), "light")
        # Directed lights give real perspective/shading without changing sensors.
        light = UsdLux.DistantLight.Define(self.stage, self.root_path + "/Lighting/Sun")
        light.CreateIntensityAttr(2200)
        light.CreateColorAttr(self.Gf.Vec3f(1, 0.90, 0.78))
        light.CreateAngleAttr(1)
        light.AddRotateXYZOp().Set(self.Gf.Vec3f(25, -35, -20))
        fill = UsdLux.DistantLight.Define(self.stage, self.root_path + "/Lighting/Fill")
        fill.CreateIntensityAttr(600)
        fill.CreateColorAttr(self.Gf.Vec3f(0.65, 0.8, 1))
        fill.AddRotateXYZOp().Set(self.Gf.Vec3f(-25, 30, 125))

    def _robot(self):
        path = self.root_path + "/TransportRobot"
        _, translation, rotation = self._group(path)
        self._box(path + "/Chassis", (0, 0, 0.30), (0.62, 0.49, 0.22), "yellow")
        self._box(path + "/Bumper", (0.33, 0, 0.27), (0.08, 0.53, 0.12), "black")
        self._box(path + "/Electronics", (0, 0, 0.44), (0.51, 0.42, 0.10), "panel")
        self._box(path + "/Deck", (-0.03, 0, 0.53), (0.46, 0.42, 0.065), "steel")
        self._box(path + "/Cargo", (-0.06, 0, 0.72), (0.35, 0.33, 0.33), "crate")
        for y in (-0.10, 0.10):
            self._box(
                path + f"/CargoStrap_{int((y + 1) * 100)}",
                (-0.06, y, 0.89),
                (0.36, 0.025, 0.012),
                "black",
            )
        for i, y in enumerate((-0.28, 0.28)):
            self._cylinder(path + f"/Tire_{i}", (0, y, 0.16), 0.16, 0.075, "black", "Y")
            self._cylinder(path + f"/Hub_{i}", (0, y * 1.17, 0.16), 0.085, 0.018, "steel", "Y")
        self._cylinder(path + "/Mast", (0.17, 0, 0.78), 0.028, 0.62, "steel")
        self._cylinder(path + "/Lidar", (0.17, 0, 1.09), 0.09, 0.08, "black")
        self._cylinder(path + "/LidarRing", (0.17, 0, 1.12), 0.091, 0.012, "cyan")
        self._box(path + "/Camera", (0.20, 0, 0.96), (0.055, 0.15, 0.08), "panel")
        self._box(path + "/CameraLens", (0.231, 0, 0.96), (0.006, 0.085, 0.037), "cyan")
        self._box(path + "/Display", (0.318, 0, 0.43), (0.008, 0.20, 0.06), "cyan")
        for y in (-0.16, 0.16):
            self._box(
                path + f"/StatusLED_{int((y + 1) * 100)}",
                (0.374, y, 0.29),
                (0.007, 0.06, 0.025),
                "cyan",
            )
        return translation, rotation

    def _human(self, i):
        """Reuse a locally pinned DynaNav skinned human as a diagnostic visual."""
        from pxr import UsdSkel, Vt, Usd

        asset = (
            Path(__file__).resolve().parents[3]
            / "third_party/TIC-VLA/rl/env/assets/peds/Collected_0006805/0006805.usd"
        )
        if not asset.is_file():
            # Explicit authored alternative when the optional upstream is absent.
            return self._authored_worker(i)
        path = f"{self.root_path}/Worker_{i}"
        _, translation, rotation = self._group(path)
        model = self.UsdGeom.Xform.Define(self.stage, path + "/Human")
        source = self.UsdGeom.Xform.Define(self.stage, path + "/Human/Asset")
        source.GetPrim().GetReferences().AddReference(str(asset))
        self.asset_manifest[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
        maps = {
            "hat_mat": "current_hat_DIFFUSE.png",
            "shoes_mat": "albedo.jpg",
            "pants10_mat": "pants10_DIFFUSE.png",
            "SMPLXmat": "People_DIFFUSE.png",
            "shirt3_mat": "shirt3_DIFFUSE.png",
        }
        for prim in Usd.PrimRange(source.GetPrim()):
            if prim.GetTypeName() != "Mesh":
                continue
            original, _ = self.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            name = original.GetPrim().GetName()
            if name not in maps:
                raise RuntimeError(f"Unknown pinned human material: {name}")
            target = "Human_" + name
            if target not in self.materials:
                self._textured_material(target, asset.parent / "textures" / maps[name])
            self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                self.materials[target], bindingStrength=self.UsdShade.Tokens.strongerThanDescendants
            )
        # Source is centimeters and Y-up; normalize to the meter/Z-up task.
        model.AddTranslateOp().Set(self.Gf.Vec3d(0, 0, -1.085))
        model.AddRotateXOp().Set(90)
        model.AddScaleOp().Set(self.Gf.Vec3d(0.0095))
        skeleton = UsdSkel.Skeleton(
            self.stage.GetPrimAtPath(path + "/Human/Asset/root/pelvis0/Skeleton")
        )
        if not skeleton:
            raise RuntimeError("Pinned DynaNav human skeleton path changed")
        translations, rotations, scales = UsdSkel.DecomposeTransforms(
            skeleton.GetRestTransformsAttr().Get()
        )
        joints = list(skeleton.GetJointsAttr().Get())
        animation = UsdSkel.Animation.Define(self.stage, path + "/Gait")
        animation.CreateJointsAttr(joints)
        animation.CreateTranslationsAttr(translations)
        animation.CreateScalesAttr(scales)
        animation.CreateRotationsAttr(rotations)
        UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
            [animation.GetPath()]
        )
        return {
            "kind": "worker",
            "translation": translation,
            "rotation": rotation,
            "swings": [],
            "animation": animation,
            "rest_rotations": list(rotations),
            "joints": joints,
            "rotation_array": Vt.QuatfArray,
            "asset": str(asset),
        }

    def _authored_worker(self, i):
        """Anatomical PPE worker with deterministic articulated walk animation."""
        path = f"{self.root_path}/Worker_{i}"
        _, translation, rotation = self._group(path)
        self._box(path + "/Torso", (0, 0, 1.20), (0.28, 0.40, 0.47), "orange")
        self._box(path + "/Waist", (0, 0, 0.94), (0.26, 0.34, 0.15), "panel")
        for y in (-0.11, 0.11):
            self._box(
                path + f"/VestStripe_{int((y + 1) * 100)}",
                (0.147, y, 1.22),
                (0.014, 0.035, 0.44),
                "white",
            )
        self._box(path + "/VestBelt", (0, 0, 1.13), (0.30, 0.41, 0.03), "white")
        head = self.UsdGeom.Sphere.Define(self.stage, path + "/Head")
        head.CreateRadiusAttr(0.115)
        head.AddTranslateOp().Set(self.Gf.Vec3d(0, 0, 1.59))
        head.AddScaleOp().Set(self.Gf.Vec3d(0.9, 0.92, 1.18))
        self._bind(head.GetPrim(), "skin")
        self._cylinder(path + "/Helmet", (0, 0, 1.73), 0.135, 0.065, "yellow")
        self._cylinder(path + "/HelmetRim", (0.025, 0, 1.70), 0.16, 0.012, "yellow")
        # X is forward; limbs swing around Y and remain separate from colliders.
        swings = []
        for j, y in enumerate((-0.11, 0.11)):
            limb = f"{path}/Leg_{j}"
            group, _, _ = self._group(limb, (0, y, 0.93))
            swing = group.AddRotateYOp()
            self._cylinder(limb + "/Trouser", (0, 0, -0.34), 0.082, 0.68, "panel")
            self._box(limb + "/Boot", (0.055, 0, -0.80), (0.26, 0.13, 0.15), "black")
            swings.append(swing)
        for j, y in enumerate((-0.27, 0.27)):
            limb = f"{path}/Arm_{j}"
            group, _, _ = self._group(limb, (0, y, 1.40))
            swing = group.AddRotateYOp()
            self._cylinder(limb + "/Sleeve", (0, 0, -0.16), 0.065, 0.32, "orange")
            self._cylinder(limb + "/Forearm", (0.025, 0, -0.43), 0.048, 0.24, "skin")
            self._box(limb + "/Glove", (0.025, 0, -0.58), (0.10, 0.10, 0.13), "black")
            swings.append(swing)
        return {
            "kind": "worker",
            "translation": translation,
            "rotation": rotation,
            "swings": swings,
        }

    def _cart(self, i):
        path = f"{self.root_path}/Cart_{i}"
        _, translation, rotation = self._group(path)
        self._box(path + "/Deck", (0, 0, 0.23), (0.90, 0.56, 0.10), "blue")
        self._box(path + "/Load", (0, 0, 0.62), (0.63, 0.45, 0.69), "crate")
        for x in (-0.33, 0.33):
            for y in (-0.24, 0.24):
                self._cylinder(
                    path + f"/Wheel_{int((x + 1) * 100)}_{int((y + 1) * 100)}",
                    (x, y, 0.13),
                    0.13,
                    0.06,
                    "black",
                    "Y",
                )
        for y in (-0.23, 0.23):
            self._cylinder(
                path + f"/HandlePost_{int((y + 1) * 100)}", (-0.40, y, 0.60), 0.025, 0.8, "steel"
            )
        self._cylinder(path + "/Handle", (-0.40, 0, 1), 0.026, 0.50, "steel", "Y")
        return {"kind": "cart", "translation": translation, "rotation": rotation, "swings": []}

    def update(self, pose, actors, time):
        self.robot[0].Set(self.Gf.Vec3d(float(pose[0]), float(pose[1]), 0))
        self.robot[1].Set(math.degrees(float(pose[2])))
        for model, actor in zip(self.models, actors):
            x, y = actor["position"]
            vx, vy = actor.get("velocity", (0, 0))
            model["translation"].Set(self.Gf.Vec3d(float(x), float(y), 0))
            if math.hypot(vx, vy) > 0.02:
                model["rotation"].Set(
                    math.degrees(math.atan2(vy, vx)) + (90 if "animation" in model else 0)
                )
            amplitude = min(1, math.hypot(vx, vy) / 0.7) * 23
            phase = math.sin(float(time) * 7)
            for j, op in enumerate(model["swings"]):
                op.Set(amplitude * phase * (1 if j in (0, 3) else -1))
            if "animation" in model:
                rotations = list(model["rest_rotations"])
                for j, name in enumerate(model["joints"]):
                    joint = str(name).split("/")[-1]
                    degrees = 0
                    axis = self.Gf.Vec3d(1, 0, 0)
                    if joint == "left_shoulder":
                        axis, degrees = self.Gf.Vec3d(0, 0, 1), -75
                    elif joint == "right_shoulder":
                        axis, degrees = self.Gf.Vec3d(0, 0, 1), 75
                    elif joint in ("left_hip", "right_hip"):
                        degrees = amplitude * phase * (1 if joint == "left_hip" else -1)
                    elif joint in ("left_knee", "right_knee"):
                        degrees = max(0, amplitude * phase * (1 if joint == "left_knee" else -1))
                    offset = self.Gf.Quatf(self.Gf.Rotation(axis, degrees).GetQuat())
                    rotations[j] = rotations[j] * offset
                model["animation"].GetRotationsAttr().Set(model["rotation_array"](rotations))

    def visible(self, value):
        self.root.GetVisibilityAttr().Set(
            self.UsdGeom.Tokens.inherited if value else self.UsdGeom.Tokens.invisible
        )

    def validate_nonphysical(self):
        """Reject any accidental collision/rigid-body schema under presentation."""
        from pxr import Usd, UsdPhysics

        for prim in Usd.PrimRange(self.root.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Presentation prim has physics: {prim.GetPath()}")
        return True
