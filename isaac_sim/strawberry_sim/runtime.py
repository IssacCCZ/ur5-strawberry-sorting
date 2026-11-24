"""Isaac Sim execution. Import only after SimulationApp has started."""
import json
import math
from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from .assets import prepare
from .importer import import_robot
from .kinematics import ARM_JOINTS, BIN_JOINTS, DROP_POSES, PREGRASP, Kinematics
from .perception import YoloDetector
from .robot_links import find_robot_link
from .scene import build_scene, world_transform


class Simulation:
    def __init__(self, app, args):
        self.app, self.args = app, args
        self.frames, self.attached = 0, None
        self.dt = 1 / 120
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        urdf, records = prepare()
        self.kinematics = Kinematics(urdf)
        self.mimic = json.loads(urdf.with_name("mimic.json").read_text())
        self.shelf_positions = {m["name"]: np.array(m["pose"][:3]) for m in records if m["name"] in DROP_POSES}
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=self.dt)
        self.stage = omni.usd.get_context().get_stage()
        fruit_paths = build_scene(self.stage, records)
        for path in fruit_paths.values():
            PhysxSchema.PhysxRigidBodyAPI.Apply(self.stage.GetPrimAtPath(path)).CreateDisableGravityAttr(True)

        self.robot_root = import_robot(self.stage, urdf)
        roots = [p for p in Usd.PrimRange(self.robot_root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one UR5 articulation, found {len(roots)}")
        self.robot = self.world.scene.add(SingleArticulation(str(roots[0].GetPath()), name="ur5"))
        self.ee = self.find_link("ee_link")
        self.wrist = self.find_link("wrist_3_link", require_rigid_body=True)
        optical = self.find_link("camera_rgb_optical_frame")
        self.fruits = {name: self.world.scene.add(SingleRigidPrim(path, name=name)) for name, path in fruit_paths.items()}

        self.camera = Camera(prim_path=str(optical.GetPath()) + "/rgbd", frequency=30, resolution=(640, 480))
        self.camera.set_local_pose(translation=np.zeros(3), orientation=np.array([1., 0., 0., 0.]), camera_axes="ros")
        # USD aperture/focal-length use the same units; preserve Gazebo's 1.04 rad horizontal FOV.
        camera_prim = UsdGeom.Camera(self.stage.GetPrimAtPath(self.camera.prim_path))
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateVerticalApertureAttr(20.955 * 480 / 640)
        camera_prim.CreateFocalLengthAttr(20.955 / (2 * math.tan(1.04 / 2)))
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 5.0))
        set_camera_view(eye=np.array([-1.85, 2.25, 1.65]), target=np.array([0.30, 0.10, 0.58]))

        self.world.reset()
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        names = list(self.robot.dof_names)
        self.arm_indices = np.array([names.index(name) for name in ARM_JOINTS], dtype=int)
        self.gripper_indices = {name: names.index(name) for name in names if name.startswith("gripper_")}
        expected_gripper = {"gripper_finger1_joint", *self.mimic}
        if not expected_gripper.issubset(self.gripper_indices):
            raise RuntimeError(f"Missing imported gripper joints: {expected_gripper - self.gripper_indices.keys()}")
        self.command = np.zeros(self.robot.num_dof)
        self.command[self.arm_indices] = PREGRASP
        self.set_gripper_command(0.10)
        self.robot.set_joint_positions(self.command)
        self.robot.set_joints_default_state(positions=self.command.copy())
        controller = self.robot.get_articulation_controller()
        kps, kds = np.full(self.robot.num_dof, 20000.), np.full(self.robot.num_dof, 1000.)
        for index in self.gripper_indices.values():
            kps[index], kds[index] = 100., 5.
        controller.set_gains(kps=kps, kds=kds)
        self.sorted = []
        self.detector = None
        self.hold(0.5)
        actual = world_transform(self.ee)
        predicted = self.kinematics.fk(self.arm_positions())
        if np.linalg.norm(actual[:3, 3] - predicted[:3, 3]) > 0.01:
            raise RuntimeError("Imported robot frame disagrees with URDF FK; refusing to execute picking motions")
        # Flatten composed layers; texture paths still refer to source assets.
        self.stage.Export(str(self.output / "strawberry_scene.usda"))
        if args.mode == "yolo":
            self.detector = YoloDetector(args.weights, args.confidence, args.device, args.vision_python)
        print(f"READY: joints={names}, mode={args.mode}, output={self.output}", flush=True)

    def find_link(self, name, *, require_rigid_body=False):
        return find_robot_link(self.robot_root, name, require_rigid_body=require_rigid_body)

    def step(self):
        if not self.app.is_running():
            raise InterruptedError("Simulation window closed")
        if self.args.max_steps and self.frames >= self.args.max_steps:
            raise TimeoutError(f"Simulation exceeded --max-steps={self.args.max_steps}")
        if self.world.is_stopped():
            raise InterruptedError("Timeline stopped; relaunch to reset the picking state")
        # Pause does not consume the finite simulation budget or advance a motion.
        while not self.world.is_playing():
            self.app.update()
            if not self.app.is_running() or self.world.is_stopped():
                raise InterruptedError("Simulation stopped while paused")
        self.robot.apply_action(ArticulationAction(joint_positions=self.command.copy()))
        self.world.step(render=self.frames % 4 == 0)
        self.frames += 1

    def hold(self, seconds):
        for _ in range(max(1, math.ceil(seconds / self.dt))):
            self.step()

    def arm_positions(self):
        values = np.asarray(self.robot.get_joint_positions())[self.arm_indices]
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Non-finite articulation state")
        return values

    def move_joints(self, target, settle=True):
        start = self.arm_positions()
        # Quintic interpolation's peak speed is 1.875 times average speed.
        duration = max(0.15, 1.875 * float(np.max(np.abs(target-start))) / self.args.speed)
        for t in np.linspace(0, 1, math.ceil(duration / self.dt) + 1)[1:]:
            blend = 10*t**3 - 15*t**4 + 6*t**5
            self.command[self.arm_indices] = start + blend * (target-start)
            self.step()
        if settle:
            self.hold(0.2)
            error = np.max(np.abs(self.arm_positions() - target))
            if error > 0.035:
                raise RuntimeError(f"Joint tracking failed: maximum error {error:.4f} rad")

    def cartesian(self, target):
        # Solve the entire segment before moving so IK failure cannot leave a half-executed segment.
        path = self.kinematics.cartesian_path(self.arm_positions(), target)
        for joints in path:
            self.move_joints(joints, settle=False)
        self.hold(0.2)
        pose = world_transform(self.ee)
        error = np.linalg.norm(pose[:3, 3] - target[:3, 3])
        if error > 0.02:
            raise RuntimeError(f"Cartesian tracking failed: end-point error {error:.4f} m")

    def set_gripper_command(self, value):
        values = {"gripper_finger1_joint": value}
        for name, entry in self.mimic.items():
            values[name] = values[entry["joint"]] * entry["multiplier"] + entry["offset"]
        for name, joint_value in values.items():
            self.command[self.gripper_indices[name]] = joint_value

    def gripper(self, value):
        self.set_gripper_command(value)
        self.hold(0.6)
        indices = np.array(list(self.gripper_indices.values()), dtype=int)
        positions = np.asarray(self.robot.get_joint_positions())[indices]
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("Gripper tracking failed: non-finite joint positions")
        error = np.max(np.abs(positions - self.command[indices]))
        if error > 0.08:
            details = {name: {"actual": round(float(position), 6),
                              "target": round(float(self.command[index]), 6)}
                       for (name, index), position in zip(self.gripper_indices.items(), positions)}
            raise RuntimeError(f"Gripper tracking failed: maximum error {error:.4f} rad; joints={details}")

    def attach(self, name):
        fruit = self.stage.GetPrimAtPath(self.fruits[name].prim_path)
        ee_position = world_transform(self.ee)[:3, 3]
        if np.linalg.norm(world_transform(fruit)[:3, 3] - ee_position) > 0.20:
            raise RuntimeError("Fruit too far from gripper to attach")
        # Preserve the current relative pose: joint creation must not snap the fruit.
        relative = np.linalg.inv(world_transform(self.wrist)) @ world_transform(fruit)
        q = Rotation.from_matrix(relative[:3, :3]).as_quat()
        joint = UsdPhysics.FixedJoint.Define(self.stage, "/World/GraspJoint")
        joint.CreateBody0Rel().SetTargets([self.wrist.GetPath()])
        joint.CreateBody1Rel().SetTargets([fruit.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*relative[:3, 3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        self.attached = name
        self.grasp_relative = relative
        self.hold(0.2)

    def check_attachment(self):
        expected = world_transform(self.wrist) @ self.grasp_relative
        actual = self.fruits[self.attached].get_world_pose()[0]
        error = np.linalg.norm(actual - expected[:3, 3])
        if error > 0.02:
            raise RuntimeError(f"Fruit did not follow the PhysX grasp joint: {error:.4f} m")

    def detach(self):
        if self.stage.GetPrimAtPath("/World/GraspJoint"):
            self.stage.RemovePrim("/World/GraspJoint")
        self.attached = None

    def capture(self, name):
        from PIL import Image
        # Give the render product fresh frames after the arm has stopped.
        self.hold(0.4)
        rgba, depth = self.camera.get_rgba(), self.camera.get_depth()
        if rgba is None or not rgba.size or depth is None or not depth.size:
            raise RuntimeError("RGB-D camera produced no data")
        Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(self.output / f"{name}.png")
        np.save(self.output / f"{name}_depth.npy", depth)
        camera_pose = world_transform(self.stage.GetPrimAtPath(self.camera.prim_path))
        # USD camera looks down -Z with +Y up; optical coordinates use +Z forward, +Y down.
        camera_pose = camera_pose @ np.diag([1., -1., -1., 1.])
        (self.output / f"{name}_camera.json").write_text(json.dumps({
            "intrinsics": self.camera.get_intrinsics_matrix().tolist(),
            "optical_to_world": camera_pose.tolist(), "depth_unit": "metre",
            "depth_type": "distance_to_image_plane"}, indent=2) + "\n")
        return rgba, depth, camera_pose

    def target(self):
        remaining = [name for name in self.fruits if name not in self.sorted]
        if self.detector is None:
            name = remaining[0]
            return name, self.fruits[name].get_world_pose()[0]
        for attempt in range(150):
            self.hold(0.2)
            camera_pose = world_transform(self.stage.GetPrimAtPath(self.camera.prim_path)) @ np.diag([1., -1., -1., 1.])
            result = self.detector.detect(self.camera.get_rgba(), self.camera.get_depth(),
                                          self.camera.get_intrinsics_matrix(), camera_pose,
                                          remaining, self.shelf_positions)
            if result is not None:
                return result
        raise TimeoutError("YOLO found no valid remaining fruit within 30 simulated seconds; inspect initial.png and depth")

    def pick(self, name, position):
        print(f"PICK: {name} at {np.asarray(position).tolist()}", flush=True)
        target = self.kinematics.fk(self.arm_positions())
        target[:3, 3] = np.asarray(position) - [0.20, 0, 0]
        self.move_joints(self.kinematics.ik(target, self.arm_positions()))
        target[0, 3] += 0.06
        self.cartesian(target)
        self.gripper(0.45)
        self.attach(name)
        target[0, 3] -= 0.10
        self.cartesian(target)
        self.check_attachment()
        self.move_joints(BIN_JOINTS)
        target = self.kinematics.fk(self.arm_positions())
        target[:3, 3] += [-0.12 if name == "strawberry1" else 0.24, 0.08, 0.0]
        self.cartesian(target)
        target[2, 3] -= 0.50
        self.cartesian(target)
        self.hold(0.3)
        self.check_attachment()
        self.detach()
        self.hold(self.dt)
        # Match the source's deterministic set_model_state placement; fruit gravity stays off.
        self.fruits[name].set_world_pose(np.array(DROP_POSES[name]), np.array([1., 0, 0, 0]))
        self.fruits[name].set_linear_velocity(np.zeros(3))
        self.fruits[name].set_angular_velocity(np.zeros(3))
        self.hold(0.3)
        if np.linalg.norm(self.fruits[name].get_world_pose()[0] - DROP_POSES[name]) > 0.02:
            raise RuntimeError(f"{name} did not remain in its bin after release")
        self.sorted.append(name)
        # Retrace the bin descent before crossing the bin rim.
        target[2, 3] += 0.50
        self.cartesian(target)
        self.move_joints(PREGRASP)
        self.gripper(0.10)
        print(f"SORTED: {name} ({len(self.sorted)}/3)", flush=True)

    def report(self, status, error=None):
        data = {"status": status, "mode": self.args.mode, "sorted": self.sorted,
                "physics_steps": self.frames, "error": error,
                "fruit_positions": {name: obj.get_world_pose()[0].tolist() for name, obj in self.fruits.items()}}
        (self.output / "result.json").write_text(json.dumps(data, indent=2) + "\n")

    def run(self):
        try:
            self.capture("initial")
            if self.args.mode == "scene":
                for _ in range(self.args.scene_steps):
                    self.step()
                self.report("scene_ready")
            else:
                while len(self.sorted) < len(self.fruits):
                    name, point = self.target()
                    self.pick(name, point)
                self.capture("final")
                self.report("completed")
        except Exception as error:
            self.detach()
            self.report("failed", str(error))
            raise
        if not self.args.headless and not self.args.exit_on_complete:
            # Closing/stopping an already completed GUI run must not overwrite its result.
            self.args.max_steps = 0
            while self.app.is_running():
                try:
                    self.step()
                except InterruptedError:
                    break
