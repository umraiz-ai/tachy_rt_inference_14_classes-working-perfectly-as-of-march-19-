# Classification Scenarios — Pipeline and Usage Guide

This document describes `classification_scenarios.py`, which runs **NPU object detection** on images and optionally applies **scenario-specific safety rules** to classify each image as **safe** or **unsafe** (or to count workers). The inference stack matches `infernce-may-28/inf_end_to_end.py` (letterbox preprocess, `frame_split` instance, `utils.yolov9.Decoder`).

Scenarios covered here:

| Scenario | Flag | Module |
|----------|------|--------|
| Plain detection | *(none)* | Built-in drawing only |
| Scaffold safety | `--scaffold_classification` | `Scenarios/scaffold/scaffold_classification.py` |
| PPE (helmet) safety | `--ppe_classification` | `Scenarios/PPE/Classification_PPE.py` |
| Worker counting | `--counting` | `Scenarios/Counting/Classification_Counting.py` |
| Heavy machine safety | `--heavy_machine_classification` | `Scenarios/Heavy_Machine/Classification_Heavy_Machine.py` |

> **Note:** The Lifted Load scenario (`--lifted_load_classification`) is implemented in the same script but is not documented here.

---

## 1. End-to-End Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Input images   │────▶│  NPU detection   │────▶│  Scenario rules     │
│  (test / safe / │     │  letterbox 416²  │     │  (optional)         │
│   unsafe dirs)  │     │  YOLOv9 decode   │     │  safe / unsafe /    │
└─────────────────┘     └──────────────────┘     │  worker count       │
                                                  └──────────┬──────────┘
                                                             │
                             ┌───────────────────────────────┴────────────────┐
                             ▼                                                ▼
                    Annotated images                              JSON results + metrics
                    (output_dir/)                               (*_results.json, *_metrics.json)
```

### Stage A — NPU lifecycle

1. **Boot** (optional) — `boot()` when `--upload_firmware true` on SPI interface
2. **Load model** — `save_model()` uploads `.tachyrt` to NPU memory
3. **Create instance** — `make_instance()` with `frame_split` algorithm
4. **Connect** — `connect_instance()` for inference handle
5. **Post-process** — `Decoder` from `infernce-may-28/utils/yolov9.py`

### Stage B — Per-image inference

For each image:

1. Read BGR image with OpenCV
2. **Letterbox** to 416×416 (INTER_AREA resize, pad 114, BGR→RGB)
3. Normalize: `(pixel - 0) / 255`, float32
4. Run `instance.process()` → raw logits
5. Decode with YOLOv9 `Decoder` → list of `[conf, class_id, x1, y1, x2, y2]` in original image coordinates
6. Filter by optional `--conf_threshold`
7. Pass detections to scenario module (or draw boxes directly)

### Stage C — Outputs

- Annotated JPEG/PNG per image under `--output_dir`
- Per-scenario JSON: `{scenario}_results.json`
- Eval metrics when using `--safe_dir` / `--unsafe_dir`:
  - `classification_metrics.json` (safe/unsafe scenarios)
  - `counting_metrics.json` (worker counting)

---

## 2. Prerequisites

```bash
export TACHY_INTERFACE=spi:host
source ~/Desktop/pyth_env/pvenv/bin/activate
cd ~/Desktop/inference__nov_migration/example
```

| Requirement | Details |
|-------------|---------|
| Hardware | Tachy-Shield NPU connected (SPI host) |
| Model | 13-class `.tachyrt` model, 416×416 input (e.g. `may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt`) |
| Firmware | Required only when `--upload_firmware true` (default path: `./firmware/tachy-shield`) |
| Python deps | `cv2`, `numpy`, `tachy_rt`, `tqdm` |

### Default model classes

The script uses a built-in 13-class label map (`DEFAULT_CLASS_DICT`):

| ID | Class | Used in scenarios |
|----|-------|-------------------|
| 0 | cement_truck | Heavy machine |
| 1 | compactor | Heavy machine |
| 2 | dump_truck | Heavy machine |
| 3 | excavator | Heavy machine |
| 4 | grader | Heavy machine |
| 5 | mobile_crane | Heavy machine |
| 6 | tower_crane | Heavy machine |
| 7 | worker | All scenarios |
| 8 | Hardhat | Scaffold, PPE, Heavy machine |
| 9 | Red_Hardhat | Scaffold, PPE, Heavy machine |
| 10 | scaffolds | Scaffold |
| 11 | Lifted Load | *(not covered here)* |
| 12 | Hook | Scaffold |

### Default decoder settings

Hardcoded in `classification_scenarios.py` (edit script to change):

| Parameter | Value |
|-----------|-------|
| Input size | 416 × 416 × 3 |
| Output shapes | `[52,52,17]`, `[26,26,17]`, `[13,13,17]` |
| `OBJ_THRESHOLD` | 0.25 |
| `NMS_THRESHOLD` | 0.2 |
| `N_CLASSES` | 13 |

---

## 3. Input Modes

### Mode A — Single folder (detection only)

Use `--test_dir` with **no scenario flag** to run plain object detection and save annotated images.

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --test_dir ./my_images \
  --output_dir ./my_output
```

### Mode B — Safe / unsafe evaluation

Use `--safe_dir` and `--unsafe_dir` **together** with **exactly one** scenario flag. Images in `safe_dir` are ground-truth **safe**; images in `unsafe_dir` are ground-truth **unsafe**.

```
Scenarios/<name>/
├── safe/          # GT: safe
│   ├── images/    # (counting only)
│   └── labels/    # (counting only — integer count per image)
└── unsafe/        # GT: unsafe
    ├── images/
    └── labels/
```

For scaffold, PPE, and heavy machine, images live directly under `safe/` and `unsafe/`. For counting, use `safe/images` + `safe/labels` (labels auto-detected as sibling `labels/` when image dir ends with `/images`).

---

## 4. Scenario Rules (Detail)

All scenarios share common detection filtering:

- Detections below the scenario confidence threshold are ignored
- Workers shorter than `max(7% × min(h,w), 64px)` are **skipped** (too small / unreliable)

---

### 4.1 Scaffold Safety (`--scaffold_classification`)

**Module:** `Scenarios/scaffold/scaffold_classification.py`

**Purpose:** Detect scaffold-related safety violations from NPU detections.

**Rules applied (image is unsafe if any rule fires):**

| Rule | Reason code | Condition |
|------|-------------|-----------|
| Helmet required | `missing_helmet` | Every eligible worker must have a hardhat or red hardhat overlapping the head region |
| Hook fastening | `missing_hook` | When scaffold is confidently detected (`max scaffold conf ≥ --scaffold_min_conf`, default 0.50), every eligible worker must have a hook whose center lies inside the worker box expanded by 35% |
| Vertical overlap | `same_vertical_area` | When scaffold is present, no two eligible workers may work on upper/lower levels simultaneously (horizontal overlap with vertical separation) |

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--scaffold_conf_threshold` | 0.30 | Min confidence for detections used in rules |
| `--scaffold_min_conf` | 0.50 | Min scaffold detection confidence to enable hook and vertical rules |
| `--skip_hook_rule` | off | Disable hook fastening rule (helmet + vertical rules still apply) |

**Example:**

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --safe_dir ./Scenarios/scaffold/safe \
  --unsafe_dir ./Scenarios/scaffold/unsafe \
  --output_dir ./Scenarios/scaffold/predictions \
  --scaffold_classification \
  --skip_hook_rule
```

---

### 4.2 PPE / Helmet Safety (`--ppe_classification`)

**Module:** `Scenarios/PPE/Classification_PPE.py`

**Purpose:** Verify that all visible workers wear helmets (hardhat or red hardhat).

**Rules:**

| Rule | Reason code | Condition |
|------|-------------|-----------|
| Helmet required | `missing_helmet` | Every eligible worker must have helmet detection in head band (top 15% of worker height, min 20 px) |
| No workers | — | Image is **safe** if no eligible workers are detected |

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ppe_conf_threshold` | 0.30 | Min confidence for worker/helmet detections |

**Example:**

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --safe_dir ./Scenarios/PPE/safe \
  --unsafe_dir ./Scenarios/PPE/unsafe \
  --output_dir ./Scenarios/PPE/predictions \
  --ppe_classification \
  --ppe_conf_threshold 0.30
```

---

### 4.3 Worker Counting (`--counting`)

**Module:** `Scenarios/Counting/Classification_Counting.py`

**Purpose:** Count eligible workers in each image and compare against ground-truth integer labels.

**Logic:**

- Count `worker` class detections above `--counting_conf_threshold`
- Skip workers below minimum height threshold
- Ground truth: one integer per image in `labels/<image_stem>.txt`

**Label file format** (single line):

```
3
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--counting_conf_threshold` | 0.30 | Min confidence for worker detections |
| `--safe_labels_dir` | auto | GT labels for safe images (default: `../labels` if safe dir is `.../images`) |
| `--unsafe_labels_dir` | auto | GT labels for unsafe images |

**Metrics** (`counting_metrics.json`):

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error \|pred − GT\| |
| RMSE | Root mean squared error |
| Exact match rate | Fraction with zero error |
| Within ±1 / ±2 | Fraction within 1 or 2 workers of GT |

**Example:**

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --safe_dir ./Scenarios/Counting/safe/images \
  --unsafe_dir ./Scenarios/Counting/unsafe/images \
  --output_dir ./Scenarios/Counting/predictions \
  --counting \
  --counting_conf_threshold 0.30
```

---

### 4.4 Heavy Machine Safety (`--heavy_machine_classification`)

**Module:** `Scenarios/Heavy_Machine/Classification_Heavy_Machine.py`

**Purpose:** Safety checks around heavy equipment (trucks, cranes, excavators, etc.).

**Detected machine classes:** cement_truck, compactor, dump_truck, excavator, grader, mobile_crane, tower_crane

**Rules (image is unsafe if any rule fires):**

| Rule | Reason code | Condition |
|------|-------------|-----------|
| Helmet near equipment | `missing_helmet` | Eligible workers **near a machine** must wear hardhat or red hardhat. Workers far from all machines are exempt |
| Signal man | `no_signal_man` | When machines are present **and** at least one worker is near equipment, a signal man (worker with red hardhat, loose match) must be present |
| Proximity | `proximity_violation` | Worker foot-to-machine distance &lt; `--heavy_machine_danger_dist` meters (default 2.0 m), computed via ground-plane homography |

**Proximity / homography:**

- By default, a **fallback ground-plane homography** is estimated from the lower image region (no separate segmentation model required on NPU)
- Worker foot = bottom-center of bounding box
- Distance measured in world coordinates (6 m × 8 m ground plane)
- Use `--heavy_machine_no_default_homography` to disable (proximity check skipped)

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--heavy_machine_conf_threshold` | 0.30 | Min confidence for detections |
| `--heavy_machine_danger_dist` | 2.0 | Proximity violation threshold (meters) |
| `--heavy_machine_no_default_homography` | off | Disable default homography (skip proximity) |
| `--skip_signal_man_rule` | off | Disable signal-man requirement |

**Example:**

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --safe_dir ./Scenarios/Heavy_Machine/safe \
  --unsafe_dir ./Scenarios/Heavy_Machine/unsafe \
  --output_dir ./Scenarios/Heavy_Machine/predictions \
  --heavy_machine_classification \
  --heavy_machine_conf_threshold 0.30
```

---

## 5. Output Files

### Annotated images

| Input mode | Output layout |
|------------|---------------|
| `--test_dir` | `output_dir/<filename>` |
| `--safe_dir` / `--unsafe_dir` | `output_dir/safe/<filename>`, `output_dir/unsafe/<filename>` |

### JSON results

**Per-image results** — `{scenario}_results.json`:

```json
{
  "file": "example.jpg",
  "status": "unsafe",
  "status_numeric": 0,
  "reasons": ["missing_helmet"],
  "subdir": "unsafe",
  "ground_truth": "unsafe",
  "ground_truth_numeric": 0,
  "correct": true
}
```

**Counting results** include `pred_count`, `gt_count`, `error`, `correct` instead of `status`.

### Classification metrics (`classification_metrics.json`)

Computed for safe/unsafe eval scenarios. Positive class = **safe**.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct rate |
| Precision / Recall / F1 | For safe class |
| Confusion matrix | TN, FP, FN, TP |

Console output example:

```
Classification metrics (positive class = safe):
  Accuracy:  0.8500
  Precision: 0.8200
  Recall:    0.8800
  F1:        0.8490
```

---

## 6. Full Command-Line Reference

### Required

| Argument | Description |
|----------|-------------|
| `--model` | Path to `.tachyrt` model file |
| `--output_dir` | Directory for annotated outputs and JSON |

### Input (one of)

| Argument | Description |
|----------|-------------|
| `--test_dir` | Flat folder of images (detection-only mode) |
| `--safe_dir` + `--unsafe_dir` | Eval folders (requires scenario flag) |

### Scenario flags (mutually exclusive)

| Flag | Scenario |
|------|----------|
| `--scaffold_classification` | Scaffold safety |
| `--ppe_classification` | PPE / helmet |
| `--counting` | Worker counting |
| `--heavy_machine_classification` | Heavy machine |

### General options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `object_detection_yolov9` | Logical name for `tachy_rt` |
| `--conf_threshold` | `None` | Extra confidence gate after NPU decode |
| `--upload_firmware` | `false` | Boot shield before inference |
| `--path_firmware` | `./firmware/tachy-shield` | Firmware directory for `boot()` |

---

## 7. Boot After Reboot or NPU Hang

If the NPU is unresponsive, boot firmware before running:

```bash
python classification_scenarios.py \
  --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
  --safe_dir ./Scenarios/PPE/safe \
  --unsafe_dir ./Scenarios/PPE/unsafe \
  --output_dir ./Scenarios/PPE/predictions \
  --ppe_classification \
  --upload_firmware true \
  --path_firmware ./firmware/tachy-shield
```

Alternatively, flush and invoke the NPU separately (see project `README.md`):

```bash
python3 flush_npu_state.py
python3 npu_invoke_example.py --model <model.tachyrt> --input_shape 416x416x3 \
  --upload_firmware true --path_firmware ./firmware/
```

---

## 8. Project Layout

```
example/
├── classification_scenarios.py          # Main entry point
├── classification_scenarios_readme.md   # This file
├── Scenarios/
│   ├── scaffold/
│   │   ├── scaffold_classification.py
│   │   ├── safe/
│   │   ├── unsafe/
│   │   └── predictions/
│   ├── PPE/
│   │   ├── Classification_PPE.py
│   │   ├── safe/
│   │   ├── unsafe/
│   │   └── predictions/
│   ├── Counting/
│   │   ├── Classification_Counting.py
│   │   ├── safe/images/ + safe/labels/
│   │   ├── unsafe/images/ + unsafe/labels/
│   │   └── predictions/
│   └── Heavy_Machine/
│       ├── Classification_Heavy_Machine.py
│       ├── safe/
│       ├── unsafe/
│       └── predictions/
└── utils/object_detection_yolov9/req_files_ppr/
    └── may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt
```

---

## 9. Troubleshooting

| Issue | Cause | Action |
|-------|-------|--------|
| `TACHY_INTERFACE is not set` | Missing env var | `export TACHY_INTERFACE=spi:host` |
| `scenario flags are mutually exclusive` | Multiple `--*_classification` flags | Use only one scenario flag per run |
| `scenario flag is required with --safe_dir/--unsafe_dir` | Eval mode without scenario | Add e.g. `--ppe_classification` |
| `Could not resolve ... role from class dict` | Model classes don't match `DEFAULT_CLASS_DICT` | Update class names in script or use matching model |
| `Label file not found` (counting) | Missing GT count txt | Add `labels/<stem>.txt` with integer count |
| NPU hang / no inference | Shield not booted | Use `--upload_firmware true` or run `npu_invoke_example.py` first |
| Poor detection quality | Threshold / model mismatch | Tune `--*_conf_threshold` or decoder thresholds in script |

---

## 10. Related Documentation

| Document | Description |
|----------|-------------|
| `infernce-may-28/Evaluatio_readme.md` | mAP evaluation on labeled test sets |
| `object_detection_yolo_coco-80cls/run_322_file.md` | Picture inference with Tachy RT 3.2.2 |
| `object_detection_yolo_coco-80cls/README__sensor.md` | Live sensor inference |
| `npu_evaluate_map_iou.py` | Combined NPU inference + IoU/mAP evaluation |
