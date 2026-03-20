# mAP and IoU Evaluation for Object Detection Model

This script evaluates an object detection model using the mean Average Precision (mAP) and Intersection over Union (IoU) metrics on a test dataset in YOLO format.

## Features

- Calculates mAP at specified IoU threshold (default: 0.5)
- Computes average IoU for all detections
- Generates precision-recall curves
- Plots IoU distribution
- Reports per-class AP values
- Measures inference performance (FPS)
- Outputs complete evaluation results in JSON format

## Usage

```bash
python evaluate_model_map_iou.py --model <model_path> --input_shape <height>x<width>x3 [--test_dir <test_dataset_dir>] [--conf_threshold <conf>] [--iou_threshold <iou>]
```

### Arguments

- `--model`: Path to the model file (required)
- `--input_shape`: Model input shape in format HxWxD, e.g., "256x416x3" (required)
- `--test_dir`: Directory containing the test dataset (default: './complete_test_set_ppe')
- `--conf_threshold`: Confidence threshold for detections (default: 0.25)
- `--iou_threshold`: IoU threshold for considering a detection as true positive (default: 0.5)

### Example

```bash
# Navigate to the inference directory first
cd /home/dpi/raspberrypi_20241209/inference

# Set the interface environment variable if not already set
export TACHY_INTERFACE=spi:host

# Run the evaluation script
python example/evaluate_model_map_iou.py \
  --model /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/oct_1_model_256x416x3_inv-f.tachyrt \
  --input_shape 256x416x3 \
  --test_dir /home/dpi/raspberrypi_20241209/inference/example/complete_test_set_ppe \
  --conf_threshold 0.25 \
  --iou_threshold 0.5
```

## Output

The script generates the following outputs:

1. Terminal output with summary statistics
2. `evaluation_results.json`: Complete evaluation results in JSON format
3. `precision_recall_curve.png`: Precision-recall curve visualization
4. `iou_distribution.png`: Distribution of IoU values

## Understanding the Results

- **mAP (mean Average Precision)**: A metric that summarizes the precision-recall curve for object detection. The higher, the better.
- **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes. Values closer to 1.0 are better.
- **AP per class**: Average Precision for each individual class.
- **FPS**: Frames per second, indicating inference speed.

## Notes

- Ensure the model's post-processing configuration is available in the expected location.
- The test dataset should be in YOLO format with images in an 'images' folder and corresponding labels in a 'labels' folder.
- Label files should use the YOLO format (class_id x_center y_center width height), with all values normalized between 0 and 1.
