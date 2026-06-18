# Object Detection (Tachy RT 3.2.2) — Sensor Input Guide

This document describes how to run object detection with live sensor input using the Tachy-Shield NPU (API 3.2.2). For picture-based inference, see `run_322_file.md`.

---

## 1. Overview

The sensor example captures frames from the Tachy-Shield camera pipeline and runs YOLOv9 object detection in real time. The primary script for API 3.2.2 is `object_detection_sen_3.2.2.py`.

Legacy wrapper scripts are also available under `bin/`:

| Mode    | Script            | Entry point                    |
|---------|-------------------|--------------------------------|
| Picture | `./bin/picture.sh` | `object_detection_pic.py`      |
| Sensor  | `./bin/sensor.sh`  | `object_detection_sen.py`      |

Example outputs:

![Picture result](./result_picture.png)
![Sensor result](./result_sensor.png)

---

## 2. Prerequisites

- Tachy RT 3.2.2 installed and `TACHY_INTERFACE` configured
- Firmware files available under `../firmware/tachy-shield` (required when `--upload_firmware true`)
- Model, class labels, and post-processing files under `req_files_ppr/` or the model subfolder
- Tachy-Shield booted and sensor pipeline active

---

## 3. Environment Configuration

```bash
export TACHY_INTERFACE=spi:host
cd ~/Desktop/inference__nov_migration/example/object_detection_yolo_coco-80cls
```

---

## 4. Running Sensor Inference (API 3.2.2)

### 4.1 Standard Run

```bash
python object_detection_sen_3.2.2.py \
  --model BSNet0-20240820_0-YOLOv9 \
  --input_shape 320x416x3 \
  --tx 2 \
  --inverse_data false \
  --inverse_sync false \
  --inverse_clock false
```

### 4.2 Run with Explicit Model and Configuration Paths

To use custom model, class, and post-processing files outside the default model subfolder layout:

```bash
python object_detection_sen_3.2.2.py \
  --model BSNet0-20240820_0-YOLOv9 \
  --input_shape 320x416x3 \
  --tx 2 \
  --inverse_data false \
  --inverse_sync false \
  --inverse_clock false \
  --model_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt \
  --class_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/class.json \
  --post_config_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json
```

When explicit paths are omitted, the script resolves defaults relative to `../utils/object_detection_yolov9/<model>/`:

- Model: `model_<input_shape>_inv-f.tachyrt`
- Classes: `class.json`
- Post-process: `post_process_<input_shape>.json`

### 4.3 Run After Reboot or NPU Hang

Boot the device and upload firmware before starting sensor inference:

```bash
python object_detection_sen_3.2.2.py \
  --model BSNet0-20240820_0-YOLOv9 \
  --input_shape 320x416x3 \
  --tx 2 \
  --inverse_data false \
  --inverse_sync false \
  --inverse_clock false \
  --upload_firmware true \
  --path_firmware ../firmware/tachy-shield
```

---

## 5. Command-Line Arguments

| Argument              | Description                                              | Default                                      |
|-----------------------|----------------------------------------------------------|----------------------------------------------|
| `--model`             | Model subfolder under `utils/object_detection_yolov9/`   | Required                                     |
| `--input_shape`       | Model input dimensions (`HxWxD`)                         | Required                                     |
| `--tx`                | Sensor TX channel (0–3)                                  | `0`                                          |
| `--inverse_data`      | Invert sensor data polarity                              | `false`                                      |
| `--inverse_sync`      | Invert sensor sync polarity                              | `false`                                      |
| `--inverse_clock`     | Invert sensor clock polarity                             | `false`                                      |
| `--upload_firmware`   | Upload firmware via `boot()` before inference              | `false`                                      |
| `--path_firmware`     | Firmware directory for `boot()`                          | `../firmware/tachy-shield`                   |
| `--model_path`        | Full path to `.tachyrt` model file                       | Resolved from `--model` and `--input_shape`  |
| `--class_path`        | Full path to `class.json`                                | Resolved from `--model`                      |
| `--post_config_path`  | Full path to `post_process_*.json`                       | Resolved from `--model` and `--input_shape`  |

---

## 6. Post-Processing Configuration

Select the post-processing JSON file that matches the model input resolution:

| Input shape   | Post-process config file              |
|---------------|---------------------------------------|
| `416x416x3`   | `post_process_416x416x3.json`         |
| `320x416x3`   | `post_process_320x416x3.json`         |
| `256x416x3`   | `post_process_256x416x3.json`         |

Update the class count in `post_process.py` (`split_logits` `n_channels`) and ensure `N_CLASSES` in the JSON file matches your model. See `run_322_file.md` for details.

---

## 7. Picture Input (Reference)

To add images for the legacy picture example (`./bin/picture.sh`):

```bash
mv <image_path>.<format> ./image/input_<N>.<format>
```

For the API 3.2.2 picture workflow, use `object_detection_pic_322.py` as described in `run_322_file.md`.

---

## 8. Related Documentation

- `run_322_file.md` — picture-based inference setup and execution
- `tachy_rt_322_api_docs.md` — Tachy RT 3.2.2 API reference (`boot`, `enable_sensor`, `make_instance`, etc.)
