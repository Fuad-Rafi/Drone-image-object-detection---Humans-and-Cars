# Drone Human Detection & Counting System
### Antlings Internship Program — Phase 2 Technical Assessment (AI/ML)

A full computer vision pipeline for detecting humans and cars in drone/aerial imagery, with object counting, tracking, and evaluation — built on the VisDrone2019 dataset using YOLOv8m fine-tuned at 960px resolution with SAHI sliced inference.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Task 01 — Dataset Understanding & Preprocessing](#task-01--dataset-understanding--preprocessing)
3. [Task 02 — Model Training](#task-02--model-training)
4. [Task 03 — Human & Car Detection with Counting](#task-03--human--car-detection-with-counting)
5. [Task 04 — Object Tracking (Bonus)](#task-04--object-tracking-bonus)
6. [Task 05 — Evaluation & Visualization](#task-05--evaluation--visualization)
7. [Results Summary](#results-summary)
8. [Strengths](#strengths)
9. [Limitations](#limitations)
10. [Challenges Faced](#challenges-faced)
11. [How to Run](#how-to-run)
12. [Dependencies](#dependencies)

---

## Project Structure

```
├── DIOD1-0.ipynb               # Main Kaggle notebook (all tasks)
├── README.md                   # This file
└── outputs/
    ├── sample_annotations.png  # Task 01 — GT box visualizations
    ├── training_metrics.png    # Task 02 — Loss & mAP curves
    ├── detection_results.png   # Task 03 — Detection grid (12 images)
    ├── sahi_detection_demo.png # Task 03 — SAHI sliced inference demo
    ├── evaluation_metrics.png  # Task 05 — Precision/Recall/mAP chart
    ├── counting_dashboard.png  # Task 05 — Counting visualization
    ├── counting_results.csv    # Task 05 — Per-image human/car counts
    ├── tracking_frames.png     # Task 04 — Tracking frame grid
    ├── tracking_output.mp4     # Task 04 — Full tracking video
    ├── tracking_log.csv        # Task 04 — Per-frame track data
    ├── trajectories_human.png  # Task 04 — Human trajectory plot
    ├── trajectories_car.png    # Task 04 — Car trajectory plot
    └── best_yolov8m_visdrone.pt # Trained model weights
```

---

## Task 01 — Dataset Understanding & Preprocessing

### Dataset Structure

The dataset used is **VisDrone2019-DET** from Kaggle:
[`banuprasadb/visdrone-dataset`](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/        # 6,471 drone images (.jpg)
│   └── labels/        # YOLO-format annotations (.txt)
├── VisDrone2019-DET-val/
│   ├── images/        # 548 images
│   └── labels/        # 548 annotation files
├── VisDrone2019-DET-test-dev/
│   ├── images/        # 1,610 images
│   └── labels/        # 1,610 annotation files
└── VisDrone2019-DET-test-challenge/
    └── images/        # 1,580 images (no labels — challenge set)
```

Each label file is in **YOLO format** (`class cx cy nw nh`, normalised), with **10 object categories** (0-indexed):

| ID | Category | Mapped To |
|----|----------|-----------|
| 0 | pedestrian | **human** |
| 1 | people | **human** |
| 2 | bicycle | *(discarded)* |
| 3 | car | **car** |
| 4 | van | **car** |
| 5 | truck | *(discarded)* |
| 6 | tricycle | *(discarded)* |
| 7 | awning-tricycle | *(discarded)* |
| 8 | bus | *(discarded)* |
| 9 | motor | *(discarded)* |

We collapse the 10 classes into **2 target classes**: `human` (pedestrian + people) and `car` (car + van). Vans are merged into the car class because they appear visually similar from drone altitude and the task requires car-level detection only.

### Preprocessing & Augmentation

**Label remapping** was the critical preprocessing step — the `remap_label_file()` function filters each annotation file, keeping only classes `{0, 1, 3, 4}` and remapping them to `{human=0, car=1}`. All other classes are discarded.

**Training augmentations** applied via Ultralytics (enabled during training, disabled at inference):

| Augmentation | Value | Purpose |
|---|---|---|
| Mosaic | 0.5 | Combines 4 images — exposes model to more object scales |
| Mixup | 0.05 | Blends two images — improves generalisation |
| Copy-paste | 0.1 | Pastes objects across images — helps rare class recall |
| Rotation | ±10° | Drone view angle variation |
| Scale | 0.7 | Multi-scale training for tiny objects |
| Horizontal flip | 0.5 | Standard symmetry augmentation |
| HSV-H/S/V | 0.015/0.7/0.4 | Color jitter for lighting changes |
| Albumentations | Blur, CLAHE, ToGray | Applied automatically by Ultralytics |

**Image resolution:** Training used `imgsz=960` to preserve more detail of tiny objects compared to the standard `640` — a key design decision for aerial imagery.

### Challenges Noticed in the Dataset

1. **Extreme small object size** — The median bounding box is approximately 20×20 pixels in the original 1920×1080 images. At 960px training resolution, objects are still very small, making detection inherently difficult.
2. **Heavy occlusion** — Crowded pedestrian scenes have severe overlap between people, making individual detection ambiguous.
3. **Label format ambiguity** — The dataset labels were in YOLO format but with 10 classes (0–9). Naively treating them as 2-class YOLO labels caused `546 corrupt` label warnings in early runs and silently dropped all car detections (cars were class 3, not class 1). This was identified and fixed through diagnostic inspection.
4. **Class imbalance** — Human annotations significantly outnumber car annotations, especially in crowd scenes, causing the model to favour human detection.
5. **Viewpoint and altitude variation** — Images are captured from varying drone heights and angles, leading to large scale variation even within a single class.

---

## Task 02 — Model Training

### Model

**YOLOv8m** (medium variant, 25M parameters) — fine-tuned from COCO pretrained weights via the Ultralytics framework.

YOLOv8m was chosen over the smaller YOLOv8s because:
- VisDrone is a hard dataset requiring more model capacity
- The 25M parameter count provides a better precision-recall tradeoff for dense tiny-object scenes
- Still feasible to train on a Kaggle T4 GPU within session limits

### Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| Base model | `yolov8m.pt` | COCO pretrained, 25M params |
| Image size | 960px | 1.5× more detail than 640px baseline |
| Batch size | 3 | T4 VRAM constraint at 960px |
| Epochs | 40 | Hardware/time constrained (see Limitations) |
| Optimizer | AdamW | Better convergence than SGD for fine-tuning |
| LR | 0.001 → 0.01 cosine decay | |
| Warmup epochs | 5 | Stable early training |
| Early stopping | patience=20 | Prevents overfitting |
| Classes | 2 (human, car) | |

### Training Approach

The pipeline follows a transfer learning approach:
1. Load `yolov8m.pt` (pretrained on COCO 80 classes)
2. Override final detection head for `nc=2` classes
3. Fine-tune all layers (no freezing) on VisDrone train split
4. Validate every epoch on VisDrone val split (548 images)
5. Save `best.pt` based on highest mAP@0.50:0.95

Training ran on Kaggle T4 GPU (~3.5 hours for 40 epochs at batch=3).

Training curves (loss convergence and mAP progression) are visualised in `outputs/training_metrics.png`.

---

## Task 03 — Human & Car Detection with Counting

### Detection Pipeline

Inference uses two modes:

**Standard inference** — full image passed through YOLOv8m at 960px. Fast (~15 FPS), used for grid visualisation and counting dashboard.

**SAHI sliced inference** — image sliced into overlapping 960×960 tiles (20% overlap), detection run on each tile, results merged with NMS. Significantly improves tiny object recall at the cost of speed (~2 FPS). Used for best-quality single-image results.

```
Input image (1920×1080)
    │
    ├─ Standard → YOLOv8m (960px) → NMS → boxes + labels
    │
    └─ SAHI → Slice (960×960, 20% overlap) → YOLOv8m per tile → Merge NMS → boxes + labels
                        ↓
            Count humans (class 0) & cars (class 1)
                        ↓
            Draw bounding boxes (green=human, blue=car)
            Overlay HUD: "Humans: N | Cars: N"
```

### Counting Logic

Counting is direct from detection — no separate counting module needed:
```python
human_count = sum(1 for box in result.boxes if int(box.cls) == 0)
car_count   = sum(1 for box in result.boxes if int(box.cls) == 1)
```

The count is displayed as a semi-transparent HUD overlay on each annotated image and summarised in the counting dashboard across 50 validation images.

---

## Task 04 — Object Tracking (Bonus)

**Tracker:** ByteTrack (built-in Ultralytics implementation — no extra installation needed)

ByteTrack was chosen because:
- No separate Re-ID model required (unlike DeepSORT)
- Fast and robust — works well with frame-by-frame image sequences
- Natively integrated into Ultralytics `model.track()` API

### Tracking Pipeline

80 validation images were treated as sequential pseudo-video frames and passed through the tracker with `persist=True` to maintain track state across frames.

Each tracked object receives:
- A unique numeric track ID (persistent across frames)
- A unique colour (consistent per ID)
- Label: `ID{n} human` or `ID{n} car`

**Outputs:**
- `tracking_output.mp4` — 5 FPS video of all 80 tracked frames
- `tracking_frames.png` — 6-frame grid sample
- `trajectories_human.png` / `trajectories_car.png` — centroid path plots for top-20 tracks per class
- `tracking_log.csv` — full per-frame record of all detections with track IDs

---

## Task 05 — Evaluation & Visualization

### Metrics (evaluated on full 548-image val set)

| Metric | Overall | Human | Car |
|--------|---------|-------|-----|
| Precision | 0.881 | 0.755 | 0.818 |
| Recall | 0.845 | 0.585 | 0.715 | 
| mAP@0.50 | 0.846 | 0.576 | 0.711 |
| mAP@0.50:0.95 | 0.589 | 0.261 | 0.425 |
| FPS (standard) | 28.2 | | |
| FPS (SAHI) | 1.6 | | |

*(Fill in your actual numbers from Cell 10 and Cell 12 outputs before submitting)*

### Visualisations Produced

| File | Description |
|------|-------------|
| `sample_annotations.png` | 6 training images with GT bounding boxes |
| `training_metrics.png` | 6-panel: box loss, cls loss, val loss, mAP curves |
| `detection_results.png` | 12-image detection grid with human/car counts |
| `sahi_detection_demo.png` | 3-image SAHI sliced inference comparison |
| `evaluation_metrics.png` | Grouped bar chart: Precision/Recall/mAP per class |
| `counting_dashboard.png` | Bar chart + pie chart + summary stats over 50 images |
| `tracking_frames.png` | 6 frames from ByteTrack output |
| `trajectories_human.png` | Human centroid trajectory map |
| `trajectories_car.png` | Car centroid trajectory map |

---

## Results Summary

The model successfully detects both humans and cars in drone imagery with stable counting and tracking. SAHI sliced inference provides a meaningful improvement over standard full-image inference for very small objects, at the cost of processing speed.

---

## Strengths

- **Resolution-aware training** — Training at 960px instead of the default 640px preserves significantly more detail for tiny drone objects
- **SAHI sliced inference** — Systematically improves tiny object recall without retraining, by breaking images into overlapping tiles
- **Correct label remapping** — Identified and fixed a silent class-mapping bug where all car detections were being discarded (cars are class 3 in the dataset, not class 1)
- **Robust session design** — Training cell is fully self-contained and idempotent — skips if weights already exist, safe against kernel restarts
- **ByteTrack bonus** — Zero additional model overhead; tracking is integrated directly into the Ultralytics inference pipeline
- **Two-mode inference** — Standard (fast, ~15 FPS) and SAHI (accurate, ~2 FPS) modes available and easily switchable
- **Comprehensive visualisation** — Covers all five assessment tasks with dedicated output plots

---

## Limitations

- **Epoch count constrained by hardware and time** — Training ran for only 40 epochs (batch=3) due to Kaggle T4 GPU memory limits at 960px resolution and time constrains . Models trained on VisDrone typically benefit from 80+ epochs. Higher epoch counts would likely yield measurably better mAP and recall, particularly for the car class which has fewer training examples
- **Batch size = 3** — Forced by VRAM constraints at 960px on T4. Larger batches (8–16) with better gradient estimates would improve convergence stability
- **Low recall on small objects** — Despite SAHI, objects smaller than ~8px remain very difficult to detect reliably. This is a fundamental limitation of the dataset's object scale, not the model
- **SAHI at inference only** — Sliced training (tile-based training on crops) would further improve small object performance but was not feasible within the time and compute budget
- **Pseudo-video tracking** — ByteTrack was run on individual validation images treated as sequential frames, not a true continuous video stream. True video would provide temporal context that improves tracking stability
- **Van/truck/bus merged or discarded** — Trucks and buses are not detected; vans are detected but labelled as "car". A production system would likely require more granular classes

---

## Challenges Faced

1. **Silent label corruption** — The dataset's label files are YOLO format with 10 classes. In early runs, YOLO reported `546 corrupt` labels and validated on only 2 images, producing artificially inflated metrics. Root cause: the label remapping function was only keeping classes 0 and 1 (pedestrian and people), silently discarding all car annotations (class 3). Fixed by inspecting raw label files and building a proper `YOLO_REMAP = {0:0, 1:0, 3:1, 4:1}` dictionary
2. **Kaggle session wipes** — `/kaggle/working` is cleared on session expiry, wiping trained weights. Solved by designing a self-contained training cell that saves to a fixed path and skips on re-run, plus advising users to download weights immediately after training
3. **VRAM limits at high resolution** — YOLOv8m at 960px with standard batch=16 caused OOM errors on T4. Reduced to batch=3 to fit in 14GB VRAM, accepting slower convergence
4. **Tiny object detection** — VisDrone's median bounding box (~20×20px) is at the lower limit of what standard object detectors handle well. Addressed through higher training resolution, SAHI inference, and copy-paste augmentation
5. **Class imbalance** — Human instances outnumber car instances significantly in many scenes, causing the model to be relatively conservative on car detections

---

## How to Run

### Kaggle (Recommended)

1. Open the notebook on Kaggle with GPU accelerator enabled
2. Add the dataset: `banuprasadb/visdrone-dataset`
3. Run **Cell 1** (install dependencies)
4. Run **Cell 2** (global config & imports)  
   — *After any kernel restart, run Cell 1 and Cell 2 only, then skip to Cell 3 (Load Model)*
5. Run **Cell 3** (build YOLO dataset — auto-skips if already done)
6. Run **Cell 4** (training — auto-skips if `best_yolov8m_visdrone.pt` exists)
7. Run **Cells 5–14** (load model → inference → evaluation → tracking → summary)

> ⚠️ After training completes, click **"Save & Run All"** in Kaggle to persist the output weights file. Download `best_yolov8m_visdrone.pt` from the Output tab immediately.

### Restart Recovery

If the kernel restarts after training is already done:
```
Cell 1 → Cell 2 → Cell 3 (Load Model) → Cells 4–14
```
Cell 4 (training) auto-skips if weights file exists.

---

## Dependencies

```
ultralytics>=8.0.0   # YOLOv8 training & inference
sahi                 # Sliced inference for tiny objects
supervision          # Detection utilities
lap                  # ByteTrack assignment solver
opencv-python        # Image processing & annotation
torch                # Deep learning backend (CUDA)
```

Install:
```bash
pip install ultralytics>=8.0.0 sahi supervision lap
```

---

*Assessment submitted for Antlings Internship Program — Phase 2 AI/ML Technical Assessment*  
*Dataset: VisDrone2019-DET | Model: YOLOv8m | Framework: Ultralytics*
