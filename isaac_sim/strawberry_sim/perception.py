"""RGB-D projection and isolated Conda inference; no ROS dependencies."""
import json
import os
import select
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .kinematics import CLASS_TO_MODEL


def project_pixel(u, v, depth, intrinsics, camera_to_world, window=5):
    if depth.ndim != 2 or not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("Depth pixel outside image")
    u, v = int(u), int(v)
    radius = window // 2
    patch = depth[max(0, v-radius):v+radius+1, max(0, u-radius):u+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 0.01) & (patch < 5.0)]
    if not valid.size:
        raise ValueError("No valid metric depth at detection center")
    z = float(np.median(valid))
    point = np.array([(u-intrinsics[0, 2])*z/intrinsics[0, 0],
                      (v-intrinsics[1, 2])*z/intrinsics[1, 1], z, 1.])
    return (camera_to_world @ point)[:3]


class YoloDetector:
    def __init__(self, weights, confidence=0.30, device="cpu", python=None):
        python = Path(python or Path.home() / "miniforge3/envs/hw2_vision/bin/python").expanduser()
        if not python.is_file():
            raise RuntimeError(f"Vision Python missing: {python}. Create vision_environment.yml or set --vision-python.")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "CONDA_PREFIX"):
            env.pop(key, None)
        env["YOLO_AUTOINSTALL"] = "false"
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        self.directory = tempfile.TemporaryDirectory(prefix="hw2-vision-")
        self.frame_path = Path(self.directory.name) / "frame.npy"
        self.process = subprocess.Popen(
            [str(python), "-u", str(Path(__file__).resolve().parents[1] / "vision_worker.py"),
             "--weights", str(Path(weights).resolve()), "--confidence", str(confidence), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        try:
            names = set(self.receive()["names"].values())
            if names != set(CLASS_TO_MODEL):
                raise ValueError(f"Unexpected classes in strawberry weights: {names}")
        except Exception:
            self.close()
            raise

    def receive(self):
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError("Conda YOLO worker did not respond within 120 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Conda YOLO worker closed its output; see the worker traceback above")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"Conda YOLO inference failed: {response['error']}")
        return response

    def infer(self, bgr):
        np.save(self.frame_path, bgr, allow_pickle=False)
        self.process.stdin.write(json.dumps({"image": str(self.frame_path)}) + "\n")
        self.process.stdin.flush()
        return self.receive()["boxes"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.directory.cleanup()

    def detect(self, rgba, depth, intrinsics, camera_to_world, remaining, shelf_positions):
        if rgba is None or depth is None or not rgba.size or not depth.size:
            return None
        # Ultralytics accepts BGR ndarray input; Isaac returns RGBA.
        bgr = np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
        for box in self.infer(bgr):
            label = box["label"]
            model = CLASS_TO_MODEL.get(label)
            if model not in remaining:
                continue
            left, top, right, bottom = box["xyxy"]
            try:
                point = project_pixel((left+right)/2, (top+bottom)/2, depth, intrinsics, camera_to_world)
            except ValueError:
                continue
            # Preserve the original shelf-workspace rejection rule.
            if np.linalg.norm(point - shelf_positions[model]) <= 0.25:
                return model, point
        return None
