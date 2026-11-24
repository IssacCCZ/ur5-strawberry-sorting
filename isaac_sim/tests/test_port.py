"""Offline regression checks using the actual project models and target poses."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from strawberry_sim.assets import prepare, resolve_uri
from strawberry_sim.kinematics import (
    ARM_JOINTS,
    BIN_JOINTS,
    DROP_POSES,
    PREGRASP,
    Kinematics,
    transform,
)
from strawberry_sim.perception import project_pixel
from strawberry_sim.scene import build_scene, world_transform, xform


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.urdf, cls.records = prepare(Path(cls.temp.name))
        cls.kinematics = Kinematics(cls.urdf)
        cls.stage = Usd.Stage.CreateInMemory()
        cls.fruit_paths = build_scene(cls.stage, cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_active_robot_description(self):
        robot = ET.parse(self.urdf).getroot()
        names = {link.get("name") for link in robot.findall("link")}
        self.assertTrue({"ee_link", "camera_rgb_optical_frame", "gripper_base_link"}.issubset(names))
        self.assertNotIn("robot_pilar", names)
        base = self.kinematics.fk(PREGRASP, "base_link")
        np.testing.assert_allclose(base[:3, 3], [0., 0., 0.36])
        self.assertFalse(robot.findall("gazebo") or robot.findall("transmission"))
        for mesh in robot.iter("mesh"):
            self.assertTrue(Path(mesh.get("filename")).is_file())
        for joint in ARM_JOINTS:
            self.assertIsNotNone(robot.find(f"joint[@name='{joint}']"))

    def test_mimic_signs_are_retained(self):
        mimic = json.loads(self.urdf.with_name("mimic.json").read_text())
        self.assertEqual(len(mimic), 5)
        self.assertEqual(mimic["gripper_finger1_finger_tip_joint"]["multiplier"], -1.)
        self.assertEqual(mimic["gripper_finger2_joint"]["multiplier"], 1.)

    def test_missing_asset_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_uri("model://not_in_workspace/mesh.dae")

    def test_scene_keeps_three_fruits_and_two_bins(self):
        self.assertEqual(set(self.fruit_paths), set(DROP_POSES))
        self.assertTrue(self.stage.GetPrimAtPath("/World/mature_bin"))
        self.assertTrue(self.stage.GetPrimAtPath("/World/other_bin"))
        for record in self.records:
            prim = self.stage.GetPrimAtPath("/World/" + record["name"])
            np.testing.assert_allclose(world_transform(prim)[:3, 3], record["pose"][:3])
            if record["name"] in DROP_POSES:
                self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))

    def test_fruit_size_and_texture_survive_conversion(self):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for name in DROP_POSES:
            prim = self.stage.GetPrimAtPath("/World/" + name)
            size = np.array(cache.ComputeWorldBound(prim).ComputeAlignedBox().GetSize())
            self.assertTrue(np.all(size > 0.02), size)
            self.assertTrue(np.all(size < 0.05), size)
            mesh = self.stage.GetPrimAtPath(f"/World/{name}/link_0/visual/mesh_0")
            self.assertTrue(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").HasValue())
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            texture = self.stage.GetPrimAtPath(str(material.GetPath()) + "/Texture").GetAttribute("inputs:file").Get()
            self.assertTrue(Path(texture.path).is_file())

    def test_world_transform_composes_rotated_parent(self):
        stage = Usd.Stage.CreateInMemory()
        xform(stage, "/parent", [1, 2, 3, 0, 0, np.pi/2])
        child = xform(stage, "/parent/child", [0.2, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(world_transform(child)[:3, 3], [1, 2.2, 3], atol=1e-8)

    def test_grasp_fixed_joint_preserves_world_pose(self):
        wrist = self.kinematics.fk(PREGRASP, "wrist_3_link")
        fruit = transform([0.501003, -0.057702, 0.890270])
        relative = np.linalg.inv(wrist) @ fruit
        np.testing.assert_allclose(wrist @ relative, fruit, atol=1e-10)
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.FixedJoint.Define(stage, "/grasp")
        joint.CreateExcludeFromArticulationAttr(True)
        self.assertTrue(joint.GetExcludeFromArticulationAttr().Get())

    def test_camera_sees_all_initial_targets(self):
        optical = self.kinematics.fk(PREGRASP, "camera_rgb_optical_frame")
        focal = 640 / (2*np.tan(1.04/2))
        intrinsics = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]])
        for record in self.records:
            if record["name"] not in DROP_POSES:
                continue
            point = (np.linalg.inv(optical) @ np.r_[record["pose"][:3], 1])[:3]
            self.assertGreater(point[2], 0.01)
            pixel = intrinsics @ point / point[2]
            self.assertTrue(0 <= pixel[0] < 640 and 0 <= pixel[1] < 480, pixel)
            depth = np.full((480, 640), point[2])
            recovered = project_pixel(pixel[0], pixel[1], depth, intrinsics, optical)
            self.assertLess(np.linalg.norm(recovered - record["pose"][:3]), 0.002)

    def test_mount_stays_behind_camera_lens(self):
        optical = self.kinematics.fk(PREGRASP, "camera_rgb_optical_frame")
        holder = self.kinematics.fk(PREGRASP, "camera_holder_link")
        link = ET.parse(self.urdf).getroot().find("link[@name='camera_holder_link']")
        for visual in link.findall("visual"):
            size = np.fromstring(visual.find("geometry/box").get("size"), sep=" ")
            origin = visual.find("origin")
            local = transform(np.fromstring(origin.get("xyz"), sep=" "),
                              np.fromstring(origin.get("rpy"), sep=" "))
            to_camera = np.linalg.inv(optical) @ holder @ local
            for corner in product((-0.5, 0.5), repeat=3):
                point = to_camera @ np.r_[np.array(corner)*size, 1.]
                self.assertLess(point[2], 0., "Mount protrudes in front of the optical plane")

    def test_all_pick_and_bin_waypoints_are_reachable(self):
        kin = self.kinematics
        for record in self.records:
            name = record["name"]
            if name not in DROP_POSES:
                continue
            with self.subTest(fruit=name):
                target = kin.fk(PREGRASP)
                target[:3, 3] = np.array(record["pose"][:3]) - [0.20, 0, 0]
                q = kin.ik(target, PREGRASP)
                for delta in ([0.06, 0, 0], [-0.10, 0, 0]):
                    target[:3, 3] += delta
                    q = kin.cartesian_path(q, target)[-1]
                    np.testing.assert_allclose(kin.fk(q), target, atol=0.002)
                q = BIN_JOINTS.copy()
                target = kin.fk(q)
                for delta in ([-0.12 if name == "strawberry1" else 0.24, 0.08, 0], [0, 0, -0.50], [0, 0, 0.50]):
                    target[:3, 3] += delta
                    q = kin.cartesian_path(q, target)[-1]
                    np.testing.assert_allclose(kin.fk(q), target, atol=0.002)

    def test_unreachable_target_does_not_return_a_motion(self):
        target = self.kinematics.fk(PREGRASP)
        target[:3, 3] = [10, 10, 10]
        with self.assertRaises(RuntimeError):
            self.kinematics.ik(target, PREGRASP)


class DepthTests(unittest.TestCase):
    def test_invalid_depth_and_out_of_bounds_are_rejected(self):
        depth = np.full((10, 10), np.nan)
        for pixel in [(5, 5), (-1, 0), (10, 0)]:
            with self.assertRaises(ValueError):
                project_pixel(*pixel, depth, np.eye(3), np.eye(4))

    def test_patch_ignores_zero_infinity_and_nan(self):
        depth = np.ones((5, 5))
        depth[2, 2], depth[2, 1], depth[1, 2] = np.nan, 0, np.inf
        intrinsics = np.array([[100, 0, 2], [0, 100, 2], [0, 0, 1]])
        np.testing.assert_allclose(project_pixel(2, 2, depth, intrinsics, transform([1, 2, 3])), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
