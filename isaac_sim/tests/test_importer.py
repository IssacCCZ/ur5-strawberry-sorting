"""Check the legacy importer contract using real USD drives and stub Kit commands."""
import importlib.util
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from pxr import Usd, UsdGeom, UsdPhysics


class TargetType(Enum):
    JOINT_DRIVE_POSITION = 1


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.root = UsdGeom.Xform.Define(self.stage, "/robot").GetPrim()
        self.config = SimpleNamespace()
        self.modules = {}
        for name in ("omni", "omni.kit", "omni.kit.commands", "isaacsim", "isaacsim.core",
                     "isaacsim.core.utils", "isaacsim.core.utils.extensions", "isaacsim.asset",
                     "isaacsim.asset.importer", "isaacsim.asset.importer.urdf"):
            module = ModuleType(name)
            self.modules[name] = module
            if "." in name:
                parent, leaf = name.rsplit(".", 1)
                setattr(self.modules[parent], leaf, module)
        self.execute = Mock(side_effect=[(True, self.config), (True, "/robot")])
        self.modules["omni.kit.commands"].execute = self.execute
        self.modules["isaacsim.core.utils.extensions"].enable_extension = Mock()
        self.modules["isaacsim.asset.importer.urdf"]._urdf = SimpleNamespace(UrdfJointTargetType=TargetType)

    def import_robot(self):
        source = Path(__file__).resolve().parents[1] / "strawberry_sim/importer.py"
        with patch.dict(sys.modules, self.modules):
            spec = importlib.util.spec_from_file_location("importer_under_test", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.import_robot(self.stage, Path("/unused/robot.urdf"))

    def drive(self, path):
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("acceleration")
        drive.CreateMaxForceAttr(0.1)
        drive.CreateTargetPositionAttr(12.0)
        return drive

    def test_legacy_drive_mode_matches_controller_without_increasing_effort(self):
        drives = [self.drive("/robot/joints/" + name) for name in
                  ("wrist_3_joint", "gripper_finger1_joint", "gripper_finger1_finger_tip_joint")]
        other = self.drive("/other_robot/joints/gripper_finger1_joint")
        self.assertEqual(self.import_robot(), self.root)
        self.assertIs(self.config.default_drive_type, TargetType.JOINT_DRIVE_POSITION)
        self.assertFalse(self.config.parse_mimic)
        for drive in drives:
            self.assertEqual(drive.GetTypeAttr().Get(), "force")
            self.assertAlmostEqual(drive.GetMaxForceAttr().Get(), 0.1)
            self.assertEqual(drive.GetTargetPositionAttr().Get(), 12.0)
        self.assertEqual(other.GetTypeAttr().Get(), "acceleration")

    def test_missing_imported_robot_is_rejected(self):
        self.execute.side_effect = [(True, self.config), (True, "/missing")]
        with self.assertRaisesRegex(RuntimeError, "missing robot prim"):
            self.import_robot()
