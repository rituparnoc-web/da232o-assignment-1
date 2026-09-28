## Object Detection Transfer Learning Assignment

### Assignment Goal

Implement and evaluate a computer vision model for object detection using a pre-trained model. The project must explain the value of pre-training and provide reproducible source code for submission through an FTP link.

### Selected Approach

- **Dataset:** Pascal VOC 2007 and 2012
- **Source training pool:** VOC 2007 `trainval` + VOC 2012 `trainval`
- **Training split:** Deterministic 90% subset of the combined training pool
- **Validation split:** Deterministic 10% subset of the combined training pool
- **Split seed:** `42`
- **Test split:** VOC 2007 `test`, kept untouched until final evaluation
- **Model:** Ultralytics YOLOv8n
- **Pre-trained checkpoint:** `yolov8n.pt`, pre-trained on COCO
- **Framework:** Python, PyTorch, and Ultralytics
- **Training method:** Transfer learning and fine-tuning
- **Deadline:** 3 October 2026

Pascal VOC is a standard object-detection benchmark with 20 object categories and bounding-box annotations. It is smaller and more practical to train than the full COCO dataset while still providing meaningful detection results.

### Purpose of the Pre-trained Model

The COCO-pre-trained model has already learned general visual features such as edges, textures, shapes, and object parts from a large and diverse dataset. Fine-tuning these features on Pascal VOC should:

- Reduce training time and data requirements.
- Improve convergence and detection accuracy.
- Provide a stronger starting point than random initialization.
- Demonstrate how knowledge learned from one dataset transfers to another.

The primary comparison will be between the original COCO-pre-trained detector and the Pascal VOC fine-tuned detector. The original model will be used mainly for qualitative baseline predictions. Any quantitative baseline evaluation will use an explicit mapping between overlapping COCO and Pascal VOC classes, with unmatched or ambiguous classes excluded and documented.

If time and computing resources permit, a secondary experiment may compare pre-trained initialization with random initialization using the same architecture and training subset. This experiment is optional and must not delay the core submission.

### Implementation Plan

#### 1. Environment and project setup

- Set up a compatible Python, PyTorch, and Ultralytics environment.
- Record exact package versions in `requirements.txt`.
- Record the model checkpoint name, source, and applicable license information.
- Set and document random seeds where supported.
- Keep dataset preparation, training, evaluation, and inference steps reproducible.

#### 2. Dataset preparation

- Download Pascal VOC 2007 and 2012 from an official or established source.
- Combine VOC 2007 `trainval` and VOC 2012 `trainval` into one source training pool.
- Create a deterministic 90:10 training-validation split using the fixed random seed `42`.
- Save the generated training and validation image lists so the exact split can be reproduced.
- Use the validation subset for training monitoring, model selection, checkpoint selection, and early stopping.
- Reserve VOC 2007 `test` strictly for one-time final evaluation and do not use it for model-development decisions.
- Convert the Pascal VOC XML annotations into the YOLO annotation format.
- Create the dataset configuration for all 20 Pascal VOC classes.
- Check for missing files, invalid boxes, and split leakage.
- Visualize sample images with their ground-truth bounding boxes.

#### 3. Pre-trained baseline

- Load the `yolov8n.pt` COCO-pre-trained detector without Pascal VOC fine-tuning.
- Run inference on a representative sample of Pascal VOC test images.
- Save representative predictions for qualitative comparison.
- Define a documented COCO-to-Pascal-VOC mapping before calculating any quantitative baseline metrics.
- Exclude unmatched or ambiguous classes from the quantitative baseline and clearly state the resulting evaluation scope.

#### 4. Fine-tuning

- Initialize YOLOv8n using the COCO-pre-trained `yolov8n.pt` weights.
- Configure the model for the 20 Pascal VOC classes.
- Fine-tune on the 90% training subset created from the combined VOC 2007 and VOC 2012 training pool.
- Monitor performance on the fixed 10% validation subset during training.
- Use validation mAP@0.5:0.95 for model selection and best-checkpoint selection.
- Apply early stopping with a documented patience value.
- Start with the standard transfer-learning configuration provided by the selected framework.
- Freeze backbone layers only if required by limited computing resources or justified by validation results.
- Save the best checkpoint, final checkpoint, training configuration, split files, and training history.

#### 5. Evaluation and analysis

- Evaluate the best fine-tuned checkpoint on the untouched VOC 2007 test split.
- Report precision, recall, mAP@0.5, and mAP@0.5:0.95 on the untouched VOC 2007 test split.
- Document the confidence threshold, non-maximum suppression settings, and IoU thresholds used during evaluation.
- Show sample detections alongside their ground-truth annotations.
- Compare the original pre-trained predictions with the fine-tuned predictions qualitatively.
- Include quantitative baseline results only if the COCO-to-VOC class mapping has been implemented and documented.
- Include representative false positives, missed detections, localization errors, and difficult cases.
- Briefly discuss class imbalance, domain differences, and model limitations.
- Report inference speed only if it can be measured consistently on documented hardware.

#### 6. Documentation

Prepare a report or notebook covering:

- Dataset selection, classes, and train/test split.
- YOLOv8n architecture and object-detection pipeline.
- Exact pre-trained checkpoint and the utility of transfer learning.
- Dataset conversion and validation steps.
- Training configuration, 90:10 split ratio, split seed, early-stopping patience, software versions, and hardware environment.
- Saved training and validation image lists for exact split reproduction.
- Validation metric used for best-checkpoint selection.
- Precision, recall, mAP@0.5, mAP@0.5:0.95, confidence threshold, non-maximum suppression settings, and IoU thresholds.
- Qualitative baseline and fine-tuned prediction comparison.
- Quantitative baseline methodology, if implemented.
- Failure cases, limitations, and possible improvements.

#### 7. Packaging and FTP submission

The final submission should contain:

- Source code.
- Dataset preparation instructions or scripts.
- Training, evaluation, and inference instructions.
- `requirements.txt` or an equivalent environment specification.
- Report or notebook.
- Best model weights, if permitted by the submission limits.
- Sample prediction images.
- A `README.md` containing the exact commands required to reproduce the results.

Before submission:

- Test the project from a clean environment where practical.
- Remove unnecessary datasets, caches, temporary files, and intermediate checkpoints.
- Verify that the FTP upload completes successfully.
- Open the FTP link independently and confirm that the required files are accessible.

### Milestones

- **16-19 September:** Set up YOLOv8n, confirm the environment, record dependencies, and finalize dataset splits.
- **20-23 September:** Prepare and validate the Pascal VOC dataset, create and save the deterministic training-validation split, then generate baseline predictions.
- **24-27 September:** Fine-tune the model and save checkpoints and training history.
- **28-30 September:** Evaluate the best model, generate comparison images, and analyze errors.
- **1-2 October:** Finalize documentation, clean the source tree, and test reproducibility.
- **3 October:** Upload the complete submission to FTP and independently verify access.

### Success Criteria

The project is complete when:

- YOLOv8n runs on Pascal VOC images using documented COCO-pre-trained weights.
- The exact dataset splits and annotation conversion process are documented.
- Fine-tuning uses the deterministic training-validation split and produces reproducible results on the selected environment.
- The best checkpoint is selected using validation mAP@0.5:0.95 and a documented early-stopping configuration.
- The selected checkpoint is evaluated once on the untouched VOC 2007 test split using precision, recall, mAP@0.5, and mAP@0.5:0.95.
- Baseline and fine-tuned predictions are compared without making invalid cross-dataset metric claims.
- The report clearly explains why pre-training is useful.
- Source code, dependencies, and reproduction commands are complete.
- The FTP submission link has been tested before the deadline.

### Optional Extensions

Complete these only after all core requirements are satisfied:

- Compare pre-trained initialization with random initialization on the same controlled subset.
- Report inference speed using fixed hardware and settings.
- Experiment with freezing and unfreezing backbone layers.
- Perform deeper per-class or class-imbalance analysis.

### Remaining Administrative Checks

- Confirm whether model weights may be included in the FTP submission.
- Confirm the required report format and any marking rubric constraints.
- Confirm whether the FTP link must be public, password-protected, or institution-accessible.
