# Pascal VOC Object Detection with YOLOv8n Transfer Learning

This project fine-tunes the COCO-pretrained Ultralytics YOLOv8n detector on Pascal VOC 2007 and 2012. The local setup was tested on an NVIDIA T1200 Laptop GPU with 4 GB VRAM, 32 GB system RAM, and an Intel i7-11800H.

## Hardware profile

These defaults were used to test the training pipeline on the local T1200:

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

Use Python 3.11 or 3.12 with the pinned dependencies:

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

These predictions are for visual inspection. I did not calculate a COCO-to-VOC baseline score because the datasets use different class definitions and annotation policies.

## Fine-tuning

```powershell
python scripts/train.py --data configs/voc.yaml --weights yolov8n.pt --epochs 100 --patience 50 --imgsz 640 --batch 2 --workers 2 --device 0
```

The best checkpoint will normally be written to:

```text
runs/detect/voc_yolov8n/weights/best.pt
```

## Final evaluation

After choosing a checkpoint using validation results, evaluate it on the held-out test split:

```powershell
python scripts/evaluate.py --weights runs/detect/voc_yolov8n/weights/best.pt --data configs/voc.yaml --imgsz 640 --batch 2 --device 0
```

The evaluation reports precision, recall, mAP@0.5, and mAP@0.5:0.95. Evaluation uses a low confidence threshold (`0.001`) so the mAP calculation can sweep confidence values. Baseline visualization uses confidence `0.25` and NMS IoU `0.7`.

## Reproducibility

The dataset split was generated with seed `42`: VOC 2007 and 2012 `trainval` images were split 90:10 for training and validation, and VOC 2007 `test` was kept for final evaluation. Training began on the local T1200 and continued from that checkpoint for 30 additional epochs on a Kaggle T4, using batch size 8 and image size 640. The Kaggle run used Python 3.12, PyTorch 2.10.0+cu128, and Ultralytics 8.3.0.

The selected checkpoint is `kaggle_results/weights/best.pt`. On the held-out test set, it reached mAP@0.5 of `0.791` and mAP@0.5:0.95 of `0.578`. Detailed metrics and plots are in `kaggle_results/eval/` and `kaggle_results/plots/`; two annotated examples are in `kaggle_results/predictions/voc_test_predictions/`. The full prediction archive and VOC dataset are not included in Git. The test results were not used to select or adjust the model.
