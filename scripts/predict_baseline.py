from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--source", type=Path, default=Path("data/processed/images/test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--name", type=str, default="coco_on_voc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        device=args.device,
        conf=0.25,
        iou=0.7,
        max_det=100,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
