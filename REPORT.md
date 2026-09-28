# Final Report: Object Detection with Transfer Learning
## Pascal VOC using YOLOv8n Pretrained on COCO

**Author:** Rituparno Chatterjee
**Deadline:** 3 October
**Submission format:** FTP source code + report

---

## 1. Assignment Objective

Implement a computer vision object detection model preceded by a pre-trained model, using a standard dataset, and explain the utility of the pre-trained model.

---

## 2. Dataset

**Pascal VOC 2007 + 2012** — a standard, widely used object detection benchmark.

| Split | Images | Source |
|---|---|---|
| Train | 14,896 | VOC2007 trainval + VOC2012 trainval (deterministic 90/10 split) |
| Validation | 1,655 | held out from combined trainval |
| Test | 4,952 | VOC2007 test (untouched until final evaluation) |

- 20 object classes (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor).
- Converted from Pascal VOC XML annotations to YOLO `.txt` label format using `scripts/prepare_dataset.py`.
- Train/validation split generated with a fixed random seed (`42`) for reproducibility; test split kept fully separate and never used during training or model selection.

---

## 3. Pre-trained Model and Its Utility

**Model:** YOLOv8n (Ultralytics), initialized with **COCO-pretrained weights** (`yolov8n.pt`).

### Why a pre-trained model was used

- COCO contains ~118,000 images across 80 classes, far larger and more diverse than Pascal VOC alone.
- The COCO-pretrained backbone has already learned general-purpose visual features (edges, textures, shapes, object parts) that transfer well to a related domain (object detection with overlapping categories, e.g. `person`, `car`, `dog`, `cat`, `bus`).
- Starting from these weights instead of random initialization:
  - Reduces the number of epochs and images needed to reach good accuracy (transfer learning).
  - Produces a stronger initial baseline, evidenced by usable detections even before any Pascal VOC fine-tuning.
  - Lowers the risk of overfitting when fine-tuning on a comparatively small dataset (~14,900 training images vs. COCO's ~118,000).
- This directly demonstrates the assignment's required use of "a pre-trained model" preceding the final detector.

### Qualitative baseline

Before fine-tuning, the unmodified `yolov8n.pt` (COCO classes) was run on 100 sample Pascal VOC test images to visually confirm the pre-trained model produces reasonable object localization, despite being trained on a different label set. Baseline predictions are available under `runs/baseline/coco_on_voc_sample`.

A quantitative COCO-vs-VOC class-level comparison was intentionally not computed, since COCO and Pascal VOC use different class taxonomies and annotation conventions; such a comparison would require an explicit class-mapping methodology that was out of scope for this assignment. Fine-tuned model results below are the primary, valid comparison point.

---

## 4. Training Configuration

| Setting | Value |
|---|---|
| Framework | Ultralytics YOLOv8, PyTorch 2.x (CUDA) |
| Model | YOLOv8n (2.69M parameters) |
| Initial weights | `yolov8n.pt` (COCO pretrained) |
| Image size | 640×640 |
| Batch size | 8 |
| Epochs | 30 (early stopping patience = 8; not triggered) |
| Optimizer | AdamW (auto-selected), lr0 ≈ 4.17e-4 |
| Seed | 42 (deterministic) |
| Mixed precision (AMP) | Enabled |
| Hardware | NVIDIA Tesla T4 (16 GB VRAM), Kaggle GPU notebook |
| Training time | 1.667 hours (30 epochs) |

### Training procedure

Training was initially started locally on an NVIDIA T1200 laptop GPU (4 GB VRAM) to validate the full pipeline (dataset preparation, YOLOv8n loading, and training loop) — this local run reached epoch 12 with sensible loss/metric trends but was too slow to complete within the deadline. The interrupted local checkpoint (`last.pt`, ~epoch 12) was then used as the starting point for continued fine-tuning on a Kaggle T4 GPU, completing a further 30 epochs (~total effective training well beyond the original single-machine budget) at a small fraction of the local time cost.

---

## 5. Results

### 5.1 Training curve (final epoch, validation split)

| Metric | Epoch 1 | Epoch 30 (final) |
|---|---|---|
| Precision | 0.661 | 0.819 |
| Recall | 0.604 | 0.667 |
| mAP@0.5 | 0.647 | 0.770 |
| mAP@0.5:0.95 | 0.441 | 0.575 |

Losses (box, classification, DFL) decreased steadily and monotonically across all 30 epochs on both training and validation splits, with no signs of divergence — training was stable throughout. Full per-epoch history: `kaggle_results/plots/results.csv`.

### 5.2 Final Test Set Evaluation (untouched, 4,952 images, 12,032 instances)

Evaluated once after training completed, using the best checkpoint selected by validation mAP. Test results were not used to modify the model.

| Metric | Value |
|---|---|
| Precision | 0.802 |
| Recall | 0.698 |
| mAP@0.5 | 0.791 |
| mAP@0.5:0.95 | 0.578 |

### 5.3 Per-class test results (mAP@0.5)

| Class | mAP@0.5 | Class | mAP@0.5 |
|---|---|---|---|
| car | 0.911 | dog | 0.824 |
| train | 0.883 | motorbike | 0.863 |
| bicycle | 0.882 | cat | 0.855 |
| bus | 0.876 | horse | 0.872 |
| person | 0.881 | tvmonitor | 0.778 |
| aeroplane | 0.842 | sheep | 0.781 |
| cow | 0.776 | diningtable | 0.768 |
| sofa | 0.756 | bird | 0.764 |
| bottle | 0.681 | boat | 0.684 |
| chair | 0.609 | pottedplant | 0.532 |

**Strongest classes:** `car`, `train`, `bicycle`, `bus`, `person` — typically large, well-represented, and visually distinct objects.

**Weakest classes:** `pottedplant`, `chair`, `bottle` — small objects, high intra-class shape variation, frequent occlusion, consistent with known difficulty patterns for these categories in Pascal VOC and COCO literature.

Full evaluation artifacts (confusion matrix, PR curves, per-class plots): `kaggle_results/eval/`.

### 5.4 Qualitative results

Sample annotated predictions on the test set (4,952 images processed): `kaggle_results/predictions/`.

---

## 6. Discussion

- The fine-tuned model substantially outperforms what would be expected from training a YOLOv8n from random initialization on ~14,900 images alone, within only 30 epochs — this improvement is attributable to the COCO-pretrained backbone's transferred features.
- Test-set metrics (mAP@0.5 = 0.791) are consistent with, and slightly higher than, validation metrics (mAP@0.5 = 0.770) at the same checkpoint, indicating the model generalizes well and is not overfitting to the validation split.
- Per-class performance gaps mirror known dataset-level challenges (small/occluded/deformable objects) rather than a training or pipeline defect.

### Limitations

- Fine-tuning was capped at 30 additional epochs on Kaggle for time-budget reasons; validation mAP was still slowly increasing at epoch 30 (no early stopping triggered), so further training would likely yield modest additional gains.
- A quantitative pre-trained-vs-fine-tuned comparison on identical classes was not performed due to the COCO/VOC class-taxonomy mismatch; only qualitative baseline predictions were generated.
- Training combined a local (T1200) partial run and a Kaggle (T4) continuation; while functionally equivalent to a single continuous run, this reflects available hardware constraints during the project.

---

## 7. Reproducibility

- **Local environment:** Python 3.12, PyTorch 2.4.1+cu124, Ultralytics 8.3.0. See `requirements.txt` and `README.md`.
- **Kaggle environment:** Python 3.12, PyTorch 2.10.0+cu128 (preinstalled), Ultralytics 8.3.0 (installed via `pip install --no-deps`).
- **Dataset preparation:** `scripts/prepare_dataset.py --voc-root data/raw/VOCdevkit --output data/processed --seed 42 --validation-ratio 0.10`
- **Local training entry point:** `scripts/train.py`
- **Kaggle training/evaluation notebook:** `kaggle_transfer_learning.ipynb` (template) / `kaggle_transfer_learning_executed.ipynb` (executed, with outputs)
- **Final model weights:** `kaggle_results/weights/best.pt`

---

## 8. Submission Contents

```text
configs/                                Dataset YAML configuration
scripts/                                Dataset prep, training, evaluation, baseline scripts
kaggle_transfer_learning.ipynb          Kaggle notebook (template)
kaggle_transfer_learning_executed.ipynb Kaggle notebook (executed, with outputs)
kaggle_results/
  weights/best.pt, last.pt              Final trained model weights
  plots/                                Training curves, confusion matrix, label distribution
  eval/                                 Test-set evaluation results and plots
  predictions/                          Annotated sample predictions on test images
requirements.txt                        Local environment dependencies
README.md                               Setup and reproduction instructions
PLAN_FINAL.md                           Project plan and methodology decisions
REPORT.md                               This report
```
