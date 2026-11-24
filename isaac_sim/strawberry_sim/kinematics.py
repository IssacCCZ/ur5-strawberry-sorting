"""URDF-based FK/IK shared by offline validation and the native controller."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
PREGRASP = np.array([0.9174927638, -2.3888567657, 2.7394921325,
                    -3.5216623804, -2.4881653693, -0.0126095816])
BIN_JOINTS = np.array([1.4295397195, -1.6110225129, 1.2382959200,
                      -1.2022184289, -1.5228190814, 1.4291107792])
CLASS_TO_MODEL = {"ripe_red": "strawberry1", "intermediate_yellow": "strawberry2", "unripe_green": "strawberry3"}
DROP_POSES = {"strawberry1": [-0.18, 0.55, 0.095], "strawberry2": [0.14, 0.55, 0.095], "strawberry3": [0.22, 0.55, 0.095]}


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


class Kinematics:
    def __init__(self, urdf):
        self.joints = {}
        self.parent_joints = {}
        root = ET.parse(urdf).getroot()
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis = joint.find("axis")
            record = {"name": joint.get("name"), "parent": joint.find("parent").get("link"),
                      "child": joint.find("child").get("link"), "type": joint.get("type"),
                      "origin": transform(xyz, rpy),
                      "axis": np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0, 0])}
            self.joints[record["name"]] = record
            self.parent_joints[record["child"]] = record
        self.lower, self.upper = [], []
        for name in ARM_JOINTS:
            limit = root.find(f"joint[@name='{name}']/limit")
            self.lower.append(float(limit.get("lower")))
            self.upper.append(float(limit.get("upper")))

    def fk(self, positions, link="ee_link"):
        values = dict(zip(ARM_JOINTS, positions)) if not isinstance(positions, dict) else positions
        chain = []
        while link in self.parent_joints:
            joint = self.parent_joints[link]
            chain.append(joint)
            link = joint["parent"]
        result = np.eye(4)
        for joint in reversed(chain):
            result = result @ joint["origin"]
            if joint["type"] in {"revolute", "continuous"}:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(joint["axis"] * values.get(joint["name"], 0.)).as_matrix()
                result = result @ motion
        return result

    def ik(self, target, seed, position_tolerance=0.002, angle_tolerance=0.02):
        if not np.all(np.isfinite(target)):
            raise ValueError("Non-finite IK target")

        def residual(q):
            actual = self.fk(q)
            return np.r_[actual[:3, 3] - target[:3, 3],
                         0.3 * Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]

        solution = least_squares(residual, np.clip(seed, self.lower, self.upper),
                                 bounds=(self.lower, self.upper), max_nfev=250,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > position_tolerance or np.linalg.norm(error[3:]) / 0.3 > angle_tolerance:
            raise RuntimeError(f"Unreachable IK target {target[:3, 3].tolist()}: residual {error.tolist()}")
        return solution.x

    def cartesian_path(self, start, target, step=0.01):
        start_pose = self.fk(start)
        distance = np.linalg.norm(target[:3, 3] - start_pose[:3, 3])
        count = max(2, int(np.ceil(distance / step)))
        # These picking segments keep a fixed end-effector orientation.
        if np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ start_pose[:3, :3].T).as_rotvec()) > 0.02:
            raise ValueError("Cartesian segment must preserve orientation")
        path, current = [], np.asarray(start)
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            pose = target.copy()
            pose[:3, 3] = (1 - alpha) * start_pose[:3, 3] + alpha * target[:3, 3]
            next_positions = self.ik(pose, current)
            if np.max(np.abs(next_positions - current)) > 0.5:
                raise RuntimeError("IK branch jump in Cartesian path")
            current = next_positions
            path.append(current)
        return path
