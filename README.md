# Pascal VOC Object Detection with YOLOv8n Transfer Learning

This project fine-tunes the COCO-pretrained Ultralytics YOLOv8n detector on Pascal VOC 2007 and 2012. It is configured for an NVIDIA T1200 Laptop GPU with 4 GB VRAM, 32 GB system RAM, and an Intel i7-11800H.

## Hardware profile

The default training profile is intentionally conservative:

- Model: `yolov8n.pt`
- Image size: 640
- Batch size: 2
- Data-loader workers: 2
- Mixed precision: enabled
- Dataset caching: disabled
- Device: CUDA device `0`
- Epochs: 100
- Early-stopping patience: 50
- Random seed: 42

If CUDA runs out of memory, use `--batch 1` or `--imgsz 512`.

## Environment setup

Use Python 3.11 or 3.12. Python 2.7 and Python 3.14 are not suitable for the pinned dependency set.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The installed NVIDIA driver supports the CUDA 12.4 PyTorch wheels used above. The local CUDA toolkit is not required for these prebuilt wheels.

## Dataset preparation

Download and extract Pascal VOC 2007 and 2012 so the directory has this structure:

```text
data/raw/VOCdevkit/VOC2007/
data/raw/VOCdevkit/VOC2012/
```

Then run:

```powershell
python scripts/prepare_dataset.py --voc-root data/raw/VOCdevkit --output data/processed --seed 42 --validation-ratio 0.10
```

The script:

- Combines VOC 2007 and VOC 2012 `trainval`.
- Creates a deterministic 90:10 train/validation split.
- Keeps VOC 2007 `test` separate for final evaluation.
- Converts XML annotations to YOLO format.
- Skips objects marked `difficult`.
- Clips boxes to image boundaries and skips invalid boxes.
- Saves `train.txt`, `val.txt`, and `test.txt`.
- Checks for duplicate image content across splits.

## Baseline predictions

Run the COCO-pretrained model on the VOC test images for qualitative baseline examples:

```powershell
python scripts/predict_baseline.py
```

This baseline is qualitative by default. It must not be treated as a directly comparable 20-class metric because COCO and Pascal VOC have different class definitions and annotation policies.

## Fine-tuning

```powershell
python scripts/train.py --data configs/voc.yaml --weights yolov8n.pt --epochs 100 --patience 50 --imgsz 640 --batch 2 --workers 2 --device 0
```

The best checkpoint will normally be written to:

```text
runs/detect/voc_yolov8n/weights/best.pt
```

## Final evaluation

Only run final test evaluation after model selection is complete:

```powershell
python scripts/evaluate.py --weights runs/detect/voc_yolov8n/weights/best.pt --data configs/voc.yaml --imgsz 640 --batch 2 --device 0
```

The evaluation reports precision, recall, mAP@0.5, and mAP@0.5:0.95. Evaluation uses a low confidence threshold (`0.001`) so the mAP calculation can sweep confidence values. Baseline visualization uses confidence `0.25` and NMS IoU `0.7`.

## Reproducibility

Record the following in the report:

- Python, PyTorch, torchvision, Ultralytics, and CUDA versions.
- GPU model and driver version.
- The saved train/validation/test image lists.
- The exact training command and output directory.
- The selected checkpoint and validation metric.
- Evaluation settings and result files.

Do not use the VOC 2007 test results to change training or model-selection decisions.
