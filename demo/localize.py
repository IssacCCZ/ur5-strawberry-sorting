"""Replay the RGB-D localization step from one recorded wrist-camera frame."""
import json
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parent / "isaac_sim"))
from strawberry_sim.perception import project_pixel


def main():
    frame = json.loads((DEMO_DIR / "wrist-frame.json").read_text())
    intrinsics = np.asarray(frame["intrinsics"])
    optical_to_world = np.asarray(frame["optical_to_world"])
    depth = np.full((frame["image_height"], frame["image_width"]), np.nan)
    print("Fruit                 u, v (px)      world x, y, z (m)")
    for detection in frame["detections"]:
        left, top, right, bottom = detection["xyxy_px"]
        u, v = int((left + right) / 2), int((top + bottom) / 2)
        # Only these 25 observed pixels are needed by the original median filter.
        depth[v - 2:v + 3, u - 2:u + 3] = detection["depth_patch_m"]
        x, y, z = project_pixel(u, v, depth, intrinsics, optical_to_world)
        print(f"{detection['label']:<20} {u:3}, {v:3}        {x:.4f}, {y:.4f}, {z:.4f}")


if __name__ == "__main__":
    main()
