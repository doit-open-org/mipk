#!/bin/sh
# Exports yolo11n.onnx (640x640, static shapes) next to this manifest.
# Needs `pip install ultralytics`; yolo11n.pt is downloaded on first use.
set -eu
cd "$(dirname -- "$0")"
python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
PY
ls -l yolo11n.onnx
