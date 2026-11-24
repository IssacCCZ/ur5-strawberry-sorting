"""Persistent Conda YOLO worker. Stdout is reserved for JSON messages."""
import argparse
import contextlib
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from ultralytics import YOLO
        model = YOLO(args.weights)
    print(json.dumps({"ready": True, "names": model.names}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close"):
                break
            bgr = np.load(request["image"], allow_pickle=False)
            with contextlib.redirect_stdout(sys.stderr):
                result = model.predict(bgr, conf=args.confidence, iou=0.5,
                                       imgsz=640, device=args.device, verbose=False)[0]
            boxes = [{"label": result.names[int(box.cls[0].item())],
                      "xyxy": box.xyxy[0].cpu().tolist(),
                      "confidence": float(box.conf[0].item())} for box in result.boxes]
            print(json.dumps({"boxes": boxes}), flush=True)
        except Exception as error:  # noqa: BLE001 - return inference failures over the worker protocol
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
