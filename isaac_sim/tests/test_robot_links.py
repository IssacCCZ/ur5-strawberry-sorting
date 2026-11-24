"""USD regression tests for duplicated link names in URDF import layouts."""
import unittest

from pxr import Usd, UsdGeom, UsdPhysics
from strawberry_sim.robot_links import find_robot_link


class RobotLinkTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.root = UsdGeom.Xform.Define(self.stage, "/robot").GetPrim()

    def prim(self, path, *, rigid=False):
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(prim)
        return prim

    def test_51_collision_and_visual_copies_are_not_link_frames(self):
        for name in ("ee_link", "wrist_3_link", "camera_rgb_optical_frame"):
            # Put the auxiliary nodes first to catch accidental first-match selection.
            self.prim(f"/robot/colliders/{name}")
            self.prim(f"/robot/visuals/{name}")
            expected = self.prim(f"/robot/{name}", rigid=True)
            self.assertEqual(find_robot_link(self.root, name), expected)

    def test_other_robot_and_scene_names_are_outside_search(self):
        self.prim("/other/ee_link", rigid=True)
        self.prim("/World/decoration/ee_link")
        expected = self.prim("/robot/ee_link", rigid=True)
        self.assertEqual(find_robot_link(self.root, "ee_link"), expected)

    def test_51_global_collision_scopes_reproduce_two_xform_matches(self):
        for name in ("ee_link", "wrist_3_link", "camera_rgb_optical_frame"):
            expected = self.prim(f"/robot/{name}", rigid=True)
            self.prim(f"/colliders/{name}")
            self.stage.DefinePrim(f"/visuals/{name}")
            old_matches = [p for p in self.stage.Traverse()
                           if p.GetName() == name and p.IsA(UsdGeom.Xform)]
            self.assertEqual(len(old_matches), 2)
            self.assertEqual(find_robot_link(self.root, name), expected)

    def test_61_nested_geometry_layout(self):
        expected = self.prim("/robot/Geometry/world/base/wrist/ee_link", rigid=True)
        self.assertEqual(find_robot_link(self.root, "ee_link"), expected)

    def test_unique_non_rigid_optical_frame_is_retained(self):
        expected = self.prim("/robot/camera/camera_rgb_optical_frame")
        self.assertEqual(find_robot_link(self.root, "camera_rgb_optical_frame"), expected)

    def test_grasp_joint_requires_a_rigid_wrist(self):
        self.prim("/robot/visuals/wrist_3_link")
        with self.assertRaisesRegex(RuntimeError, "require_rigid_body=True"):
            find_robot_link(self.root, "wrist_3_link", require_rigid_body=True)

    def test_ambiguous_links_report_paths_instead_of_picking_one(self):
        for rigid in (False, True):
            with self.subTest(rigid=rigid):
                a = self.prim(f"/robot/a{rigid}/ambiguous", rigid=rigid)
                b = self.prim(f"/robot/b{rigid}/ambiguous", rigid=rigid)
                with self.assertRaises(RuntimeError) as error:
                    find_robot_link(self.root, "ambiguous")
                self.assertIn(str(a.GetPath()), str(error.exception))
                self.assertIn(str(b.GetPath()), str(error.exception))

    def test_missing_link_does_not_use_another_robot(self):
        self.prim("/other/ee_link", rigid=True)
        with self.assertRaisesRegex(RuntimeError, "candidates: none"):
            find_robot_link(self.root, "ee_link")
