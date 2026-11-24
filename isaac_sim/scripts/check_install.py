"""Check the installed vision stack and original weights without starting Isaac."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision', choices=('cuda', 'cpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import dill
    import numpy as np
    import torch
    import torchvision
    import ultralytics

    expected = {'numpy': '1.26.4', 'opencv': '4.11.0', 'ultralytics': '8.3.203', 'dill': '0.4.1'}
    actual = {'numpy': np.__version__, 'opencv': cv2.__version__,
              'ultralytics': ultralytics.__version__, 'dill': dill.__version__}
    if actual != expected:
        raise RuntimeError(f'Unexpected package versions: {actual}; expected {expected}')
    flavor = 'cu128' if args.vision == 'cuda' else 'cpu'
    if torch.__version__ != f'2.7.1+{flavor}' or torchvision.__version__ != f'0.22.1+{flavor}':
        raise RuntimeError(f'Unexpected torch / torchvision: {torch.__version__}, {torchvision.__version__}')
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Vision environment must use Python 3.11')
    device = 'cpu'
    gpu = None
    if args.vision == 'cuda':
        if torch.version.cuda != '12.8' or not torch.cuda.is_available():
            raise RuntimeError('PyTorch CUDA 12.8 is unavailable; check NVIDIA driver and reboot state')
        device = 'cuda:0'
        gpu = torch.cuda.get_device_name(0)
        # Force a real kernel launch, rather than trusting the availability flag alone.
        value = torch.eye(32, device=device)
        torch.testing.assert_close(value @ value, value)
        torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    weights = root / 'src/code/Yolov8_ros/yolov8_ros/weights/strawberry_best.pt'
    model = ultralytics.YOLO(weights)
    if set(model.names.values()) != {'ripe_red', 'intermediate_yellow', 'unripe_green'}:
        raise RuntimeError(f'Unexpected classes: {model.names}')
    frame = cv2.cvtColor(np.zeros((480, 640, 4), dtype=np.uint8), cv2.COLOR_RGBA2BGR)
    model.predict(frame, device=device, imgsz=640, verbose=False)
    report = {'status': 'passed', 'python': sys.version, 'executable': sys.executable,
              **actual, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'torch_cuda': torch.version.cuda, 'gpu': gpu, 'weights': str(weights),
              'original_weights_inference': 'passed', 'device': device}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
