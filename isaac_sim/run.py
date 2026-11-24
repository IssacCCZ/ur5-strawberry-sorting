#!/usr/bin/env python3
"""Standalone entry point; --prepare-only can run without Isaac Sim."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="UR5 strawberry sorting — Isaac Sim")
    parser.add_argument("--prepare-only", action="store_true", help="expand Xacro and validate local assets without Isaac")
    parser.add_argument("--mode", choices=["scene", "ground-truth", "yolo"], default="ground-truth")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-on-complete", action="store_true")
    parser.add_argument("--scene-steps", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=120000, help="finite physics-step budget; 0 disables")
    parser.add_argument("--speed", type=float, default=0.8, help="maximum commanded arm joint speed in rad/s")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--vision-python", type=Path, help="Conda vision Python; defaults to ~/miniforge3/envs/hw2_vision/bin/python")
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[1] / "src/code/Yolov8_ros/yolov8_ros/weights/strawberry_best.pt")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    if not 0 < args.speed <= 2 or not 0 < args.confidence <= 1 or args.max_steps < 0 or args.scene_steps < 0:
        parser.error("speed must be in (0,2], confidence in (0,1], step counts nonnegative")
    if args.prepare_only:
        from strawberry_sim.assets import prepare
        robot, scene = prepare()
        print(json.dumps({"urdf": str(robot), "models": [m["name"] for m in scene]}, indent=2))
        return 0
    if args.mode == "yolo" and not args.weights.is_file():
        parser.error(f"YOLO weights missing: {args.weights}")
    try:
        from isaacsim import SimulationApp
    except ImportError:
        parser.exit(2, "Isaac Sim Python is unavailable. Run with ISAAC_SIM_PATH=/path/to/isaac-sim ./isaac_sim/run.sh\n")
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    simulation = None
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("isaacsim.core.api", "isaacsim.sensors.camera"):
            manager.set_extension_enabled_immediate(extension, True)
        from strawberry_sim.runtime import Simulation
        simulation = Simulation(app, args)
        simulation.run()
    except Exception as error:
        import traceback
        traceback.print_exc()
        if simulation is None:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "result.json").write_text(json.dumps({
                "status": "initialization_failed", "mode": args.mode,
                "sorted": [], "error": str(error)}, indent=2) + "\n")
        raise
    finally:
        if simulation is not None and simulation.detector is not None:
            simulation.detector.close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
