from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    return parser.parse_args()


def read_split(voc_dir: Path, split_name: str) -> list[str]:
    split_file = voc_dir / "ImageSets" / "Main" / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def convert_annotation(annotation_path: Path, image_path: Path) -> list[str]:
    root = ET.parse(annotation_path).getroot()
    width = float(root.findtext("size/width", "0"))
    height = float(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {annotation_path}")

    labels = []
    for object_node in root.findall("object"):
        class_name = object_node.findtext("name", "").strip().lower()
        if class_name not in CLASS_TO_INDEX:
            continue
        if object_node.findtext("difficult", "0").strip() == "1":
            continue
        box = object_node.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        xmin = min(max(xmin, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        xmax = min(max(xmax, 0.0), width)
        ymax = min(max(ymax, 0.0), height)
        if xmax <= xmin or ymax <= ymin:
            continue
        center_x = ((xmin + xmax) / 2.0) / width
        center_y = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(
            f"{CLASS_TO_INDEX[class_name]} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return labels


def copy_record(voc_dir: Path, image_id: str, destination: Path, prefix: str) -> str:
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    image_path = voc_dir / "JPEGImages" / f"{image_id}.jpg"
    annotation_path = voc_dir / "Annotations" / f"{image_id}.xml"
    if not image_path.exists() or not annotation_path.exists():
        raise FileNotFoundError(f"Missing image or annotation for {voc_dir.name}/{image_id}")
    output_stem = f"{prefix}_{image_id}"
    shutil.copy2(image_path, destination / "images" / f"{output_stem}.jpg")
    labels = convert_annotation(annotation_path, image_path)
    (destination / "labels" / f"{output_stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
    return output_stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    vocdevkit = args.voc_root
    voc2007 = vocdevkit / "VOC2007"
    voc2012 = vocdevkit / "VOC2012"
    if not voc2007.exists() or not voc2012.exists():
        raise FileNotFoundError("--voc-root must contain VOC2007 and VOC2012 directories")

    output = args.output
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    records = [(voc2007, image_id, "VOC2007") for image_id in read_split(voc2007, "trainval")]
    records += [(voc2012, image_id, "VOC2012") for image_id in read_split(voc2012, "trainval")]
    unique_records = {(voc_dir.name, image_id): (voc_dir, image_id, prefix) for voc_dir, image_id, prefix in records}
    records = list(unique_records.values())
    random.Random(args.seed).shuffle(records)
    validation_count = round(len(records) * args.validation_ratio)
    validation_records = records[:validation_count]
    training_records = records[validation_count:]

    train_list = [copy_record(voc_dir, image_id, output / "train", prefix) for voc_dir, image_id, prefix in training_records]
    val_list = [copy_record(voc_dir, image_id, output / "val", prefix) for voc_dir, image_id, prefix in validation_records]
    test_records = [(voc2007, image_id, "VOC2007") for image_id in read_split(voc2007, "test")]
    test_list = [copy_record(voc_dir, image_id, output / "test", prefix) for voc_dir, image_id, prefix in test_records]

    train_hashes = {sha256(path) for path in (output / "images" / "train").glob("*.jpg")}
    val_hashes = {sha256(path) for path in (output / "images" / "val").glob("*.jpg")}
    test_hashes = {sha256(path) for path in (output / "images" / "test").glob("*.jpg")}
    if train_hashes & test_hashes or val_hashes & test_hashes or train_hashes & val_hashes:
        raise RuntimeError("Duplicate image content detected across dataset splits")

    (output / "train.txt").write_text("\n".join(train_list) + "\n")
    (output / "val.txt").write_text("\n".join(val_list) + "\n")
    (output / "test.txt").write_text("\n".join(test_list) + "\n")
    print(f"Prepared {len(train_list)} train, {len(val_list)} validation, and {len(test_list)} test images")


if __name__ == "__main__":
    main()


