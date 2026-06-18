# Object Detection (Tachy RT 3.2.2) — Setup and Execution Guide

This document describes how to configure and run `object_detection_pic_322.py` for picture-based object detection using the Tachy-Shield NPU (API 3.2.2).

---

## 1. Prerequisites

- Tachy RT 3.2.2 environment configured on the host
- Model, class labels, and post-processing files under `req_files_ppr/`
- Input images named `input*` placed in the target input directory

---

## 2. Environment Configuration

Set the Tachy interface and change to the example directory:

```bash
export TACHY_INTERFACE=spi:host
cd ~/Desktop/inference__nov_migration/example/object_detection_yolo_coco-80cls
```

---

## 3. Script Configuration

Open `object_detection_pic_322.py` and update the following paths and parameters as required.

### 3.1 Model Path

```python
args.model_path = '/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt'
```

### 3.2 Class Labels

```python
args.clss_dict = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/class.json')
```

### 3.3 Post-Processing Configuration

Select the post-processing JSON file that matches the model input resolution:

```python
post_config = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json')
```

Available configurations:

| Input shape   | Post-process config file              |
|---------------|---------------------------------------|
| `416x416x3`   | `post_process_416x416x3.json`         |
| `320x416x3`   | `post_process_320x416x3.json`         |
| `256x416x3`   | `post_process_256x416x3.json`         |

---

## 4. Post-Processing Module

Edit the class count in:

`/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process.py`

Update `n_channels` in `split_logits` to match the number of classes in your model. The second value is the class count (e.g. 13 for a 13-class model, 80 for COCO):

```python
def split_logits(self, x, n, n_channels=(4, 13)):  # e.g. (4, 80) for COCO 80-class
```

Ensure that `N_CLASSES` in the corresponding `post_process_*.json` file matches this value.

---

## 5. Running Inference

### 5.1 Standard Run

```bash
python object_detection_pic_322.py \
  --input_shape 416x416x3 \
  --input_dir ../PPE_Ladder_hat/nipa_examples/
```

Replace `--input_dir` with the directory containing your `input*` images.

### 5.2 Run After Reboot or NPU Hang

If the NPU has been rebooted or is unresponsive, boot the device first:

```bash
python npu_invoke_example.py
```

Then run inference with firmware upload enabled:

```bash
python object_detection_pic_322.py \
  --input_shape 416x416x3 \
  --input_dir ./image \
  --upload_firmware true \
  --path_firmware ../firmware
```

---

## 6. Detection Thresholds

To obtain simpler, more visible detections on example images, adjust the following values in the appropriate `post_process_*.json` file:

```json
"OBJ_THRESHOLD": 0.25,
"NMS_THRESHOLD": 0.2
```

Lower `OBJ_THRESHOLD` values increase sensitivity; lower `NMS_THRESHOLD` values allow more overlapping boxes to be retained.
