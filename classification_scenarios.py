#!/usr/bin/env python
# coding: utf-8

"""
This script is used to run object detection inference on a directory of images and save annotated outputs.
It can also be used to run scaffold classification on the images.

Usage:
python classification_scenarios.py --model <model_path> --model_name <model_name> --input_shape <input_shape> --post_process_config <post_process_config> --post_process_module <post_process_module> --class_json <class_json> --test_dir <test_dir> --output_dir <output_dir> --conf_threshold <conf_threshold> --scaffold_classification <scaffold_classification> --scaffold_conf_threshold <scaffold_conf_threshold> --upload_firmware <upload_firmware> --path_firmware <path_firmware>

Example:
python classification_scenarios.py --model /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/model.tachyrt --model_name object_detection_yolov9 --input_shape 416x416x3 --post_process_config /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json --post_process_module /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process.py --class_json /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/class.json --test_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/test_dir --output_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/output_dir --conf_threshold 0.25 --scaffold_classification true --scaffold_conf_threshold 0.30 --upload_firmware true --path_firmware /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/firmware

Example:
python classification_scenarios.py --model /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/model.tachyrt --model_name object_detection_yolov9 --input_shape 416x416x3 --post_process_config /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json --post_process_module /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process.py --class_json /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/class.json --test_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/test_dir --output_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/output_dir --conf_threshold 0.25 --scaffold_classification true --scaffold_conf_threshold 0.30 --upload_firmware true --path_firmware /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/firmware

I can now run the script with the following command on the scaffold classification two directories safe and unsafe
one for safe and one for unsafe and store the results in the output directory and print the metrics and save the results in a json file

here is the command:
python classification_scenarios.py --model /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/model.tachyrt --model_name object_detection_yolov9 --input_shape 416x416x3 --post_process_config /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json --post_process_module /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process.py --class_json /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/class.json --safe_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/safe_dir --unsafe_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/unsafe_dir --output_dir /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/output_dir --conf_threshold 0.25 --scaffold_classification true --scaffold_conf_threshold 0.30 --upload_firmware true --path_firmware /home/dpi/raspberrypi_20241209/inference/example/utils/object_detection_yolov9/req_files_ppr/firmware

"""

import os
import sys
import cv2
import glob
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
import tachy_rt.core.functions as rt_core

sys.path.append('./utils/common')
from functions import read_json

_scaffold_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'scaffold')
if _scaffold_dir not in sys.path:
    sys.path.append(_scaffold_dir)
from scaffold_classification import detect_scaffold, resolve_scaffold_class_ids


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run object detection inference on a directory of images and save annotated outputs'
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the .tachyrt model file')

    parser.add_argument('--model_name', type=str, default="object_detection_yolov9",
                        help='Logical model name used by tachy_rt')

    parser.add_argument('--input_shape', type=str, required=True,
                        help='Model input shape (HxWxD)')

    parser.add_argument('--post_process_config', type=str, required=True,
                        help='Path to the post-process JSON config')

    parser.add_argument('--post_process_module', type=str, required=True,
                        help='Path to post_process.py containing the Decoder class')

    parser.add_argument('--class_json', type=str, required=True,
                        help='Path to class.json (id -> name mapping)')

    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory containing input images (flat folder). Mutually exclusive with --safe_dir/--unsafe_dir.')

    parser.add_argument('--safe_dir', type=str, default=None,
                        help='Ground-truth SAFE images (class 1). Must be used with --unsafe_dir.')

    parser.add_argument('--unsafe_dir', type=str, default=None,
                        help='Ground-truth UNSAFE images (class 2). Must be used with --safe_dir.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write annotated output images')

    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='Optional extra confidence gate applied after post-process')

    parser.add_argument(
        '--scaffold_classification',
        action='store_true',
        help='Apply scaffold safe/unsafe rules on NPU detections instead of plain box labels',
    )

    parser.add_argument(
        '--scaffold_conf_threshold',
        type=float,
        default=0.30,
        help='Confidence gate for scaffold rule logic (default: 0.30)',
    )

    parser.add_argument('--upload_firmware', type=str, default='false',
                        help='Upload firmware when using spi interface (true/false)')

    parser.add_argument('--path_firmware', type=str, default='./firmware/tachy-shield',
                        help='Firmware directory path for tachy_rt 3.2.2 boot()')

    args = parser.parse_args()

    if "TACHY_INTERFACE" not in os.environ:
        print('Environment variable "TACHY_INTERFACE" is not set')
        exit()

    args.interface = os.environ["TACHY_INTERFACE"]
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))
    args.upload_firmware = args.upload_firmware.lower() == 'true'
    args.clss_dict = read_json(args.class_json)
    os.makedirs(args.output_dir, exist_ok=True)

    has_test_dir = args.test_dir is not None
    has_safe = args.safe_dir is not None
    has_unsafe = args.unsafe_dir is not None

    if has_safe != has_unsafe:
        print("Error: --safe_dir and --unsafe_dir must be provided together.")
        exit(1)

    if has_test_dir and (has_safe or has_unsafe):
        print("Error: use either --test_dir OR (--safe_dir and --unsafe_dir), not both.")
        exit(1)

    if not has_test_dir and not (has_safe and has_unsafe):
        print("Error: provide --test_dir, or both --safe_dir and --unsafe_dir.")
        exit(1)

    args.eval_mode = has_safe and has_unsafe

    if args.eval_mode and not args.scaffold_classification:
        print("Error: --scaffold_classification is required when using --safe_dir/--unsafe_dir.")
        exit(1)

    if args.scaffold_classification or args.eval_mode:
        args.scaffold_class_ids = resolve_scaffold_class_ids(args.clss_dict)
        print(f"Scaffold class IDs: {args.scaffold_class_ids}")

    return args


IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def collect_image_files(directory):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def build_image_jobs(args):
    if args.eval_mode:
        jobs = []
        for path in collect_image_files(args.safe_dir):
            jobs.append({'path': path, 'gt': 1, 'subdir': 'safe', 'gt_label': 'safe'})
        for path in collect_image_files(args.unsafe_dir):
            jobs.append({'path': path, 'gt': 0, 'subdir': 'unsafe', 'gt_label': 'unsafe'})
        return jobs

    return [
        {'path': path, 'gt': None, 'subdir': '', 'gt_label': None}
        for path in collect_image_files(args.test_dir)
    ]


def compute_classification_metrics(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return None

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_safe = sum(1 for t in y_true if t == 1)
    n_unsafe = sum(1 for t in y_true if t == 0)
    safe_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    unsafe_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positive_class': 'safe',
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'n_safe': n_safe,
        'n_unsafe': n_unsafe,
        'safe_correct': safe_correct,
        'unsafe_correct': unsafe_correct,
    }


def print_classification_metrics(metrics):
    cm = metrics['confusion_matrix']
    print("\nClassification metrics (positive class = safe):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("  Confusion matrix (rows=GT, cols=Pred):")
    print("                pred_unsafe  pred_safe")
    print(f"    GT_unsafe      {cm['tn']:4d}       {cm['fp']:4d}")
    print(f"    GT_safe        {cm['fn']:4d}       {cm['tp']:4d}")
    print(f"  Safe images:   {metrics['n_safe']:4d}  (correct: {metrics['safe_correct']})")
    print(f"  Unsafe images: {metrics['n_unsafe']:4d}  (correct: {metrics['unsafe_correct']})")


def _build_boot_data(path_firmware: str):
    spl = os.path.join(path_firmware, "spl.bin")
    uboot = os.path.join(path_firmware, "u-boot.bin")
    if not os.path.exists(uboot):
        alt = os.path.join(path_firmware, "uboot.bin")
        if os.path.exists(alt):
            uboot = alt

    kernel = os.path.join(path_firmware, "image.ub")

    fpga = os.path.join(path_firmware, "fpga_top.bin")
    if not os.path.exists(fpga):
        alt = os.path.join(path_firmware, "fpga.bin")
        if os.path.exists(alt):
            fpga = alt

    required = [spl, uboot, kernel, fpga]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing firmware files for tachy_rt 3.2.2 boot():")
        for m in missing:
            print(" -", m)
        return None

    return {
        "spl": {"path": spl, "addr": "0x0"},
        "uboot": {"path": uboot, "addr": "0x2000_0000"},
        "kernel": {"path": kernel, "addr": "0x4000_0000"},
        "fpga": {"path": fpga, "addr": "0x3000_0000"},
    }


def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return True

    spi_type = args.interface.split(":")[-1]
    data = _build_boot_data(args.path_firmware)
    if data is None:
        return False

    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        print("Success to boot. Check the status via uart or other api")
        return True

    print("Failed to boot")
    print("Error code :", rt_core.get_last_error_code())
    return False


def save_model(args):
    ret = rt_core.save_model(
        args.interface, args.model_name, rt_core.MODEL_STORAGE_MEMORY, args.model, overwrite=True
    )
    return ret


def make_instance(args):
    args.instance_name = f"{args.model_name}_inst"
    try:
        rt_core.deinit_instance(args.interface, args.instance_name)
    except Exception:
        pass
    try:
        rt_core.deinit_instance(args.interface, args.model_name)
    except Exception:
        pass

    args.config = {
        "global": {
            "name": args.model_name,
            "data_type": rt_core.DTYPE_FLOAT16,
            "buf_num": 5,
            "max_batch": 1,
            "npu_mask": -1
        },
        "input": [
            {
                "method": rt_core.INPUT_FMT_BINARY,
                "std": 255.0,
                "mean": 0.0
            }
        ],
        "output": {
            "reorder": True
        }
    }

    for algo in ("frame_spliter", "frame_splitter"):
        ret = rt_core.make_instance(
            args.interface,
            args.model_name,
            args.instance_name,
            algo,
            args.config
        )
        if ret:
            print(f"make_instance success with algorithm: {algo}")
            return True

    print("make_instance fail")
    print("Error :", rt_core.get_last_error_code())
    return False


def connect_instance(args):
    ret, args.instance = rt_core.connect_instance(args.interface, args.instance_name)
    if not ret:
        print("Connect instance fail")
        print("Error :", rt_core.get_last_error_code())
        return False
    return ret


def load_post_processor(args):
    if not os.path.isfile(args.post_process_config):
        print(f"Post-process config not found: {args.post_process_config}")
        exit(-1)

    args.post_config = read_json(args.post_process_config)
    print(f"Using post-processing config: {args.post_process_config}")

    if not os.path.isfile(args.post_process_module):
        print(f"Post-process module not found: {args.post_process_module}")
        exit(-1)

    post_module_dir = os.path.dirname(os.path.abspath(args.post_process_module))
    if post_module_dir not in sys.path:
        sys.path.append(post_module_dir)
    args.post = __import__('post_process').Decoder(args.post_config)
    print(f"Using post-processing module: {args.post_process_module}")


def npu_detect(args, orig):
    orig_h, orig_w = orig.shape[:2]
    sx = orig_w / args.w
    sy = orig_h / args.h

    resized = cv2.resize(orig, (args.w, args.h))
    image = resized.reshape(-1, args.h, args.w, 3)

    args.instance.process([[image]])
    ret = args.instance.get_result()

    detected_boxes = args.post.main(
        ret['buf'].view(np.float32),
        np.array([[0, 0, args.w - 1, args.h - 1]], dtype=np.float32)
    )

    detections = []
    for box in detected_boxes:
        if len(box) < 6:
            continue
        confidence = float(box[0])
        if args.conf_threshold is not None and confidence < args.conf_threshold:
            continue
        class_id = int(box[1])
        x1, y1, x2, y2 = map(float, box[2:6])
        detections.append({
            'class_id': class_id,
            'confidence': confidence,
            'box': [
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
            ],
        })
    return detections


def draw_detections(image, detections, clss_dict):
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_name = clss_dict.get(str(det['class_id']), f"Class_{det['class_id']}")
        label = f"{class_name} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            annotated, label, (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    return annotated


def run_inference(args):
    image_jobs = build_image_jobs(args)
    num_images = len(image_jobs)

    if num_images == 0:
        if args.eval_mode:
            print(f"No images found in {args.safe_dir} or {args.unsafe_dir}")
        else:
            print(f"No images found in {args.test_dir}")
        return

    if args.eval_mode:
        os.makedirs(os.path.join(args.output_dir, 'safe'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'unsafe'), exist_ok=True)
        mode = "scaffold eval (safe/unsafe)"
    else:
        mode = "scaffold classification" if args.scaffold_classification else "detection"

    print(f"Processing {num_images} images ({mode})...")
    start_time = time.time()
    total_inference_time = 0
    saved_count = 0
    scaffold_results = []
    pred_safe_count = 0
    pred_unsafe_count = 0
    y_true = []
    y_pred = []

    for job in tqdm(image_jobs):
        image_file = job['path']
        orig = cv2.imread(image_file)
        if orig is None:
            print(f"Could not read image: {image_file}")
            continue

        inference_start = time.time()
        detections = npu_detect(args, orig)
        total_inference_time += time.time() - inference_start

        if args.scaffold_classification:
            annotated, status_numeric, reasons = detect_scaffold(
                orig,
                detections,
                args.scaffold_class_ids,
                conf_threshold=args.scaffold_conf_threshold,
            )
            status = "safe" if status_numeric == 1 else "unsafe"
            if status_numeric == 1:
                pred_safe_count += 1
            else:
                pred_unsafe_count += 1

            result_entry = {
                'file': os.path.basename(image_file),
                'status': status,
                'status_numeric': status_numeric,
                'reasons': reasons,
            }
            if args.eval_mode:
                result_entry.update({
                    'subdir': job['subdir'],
                    'ground_truth': job['gt_label'],
                    'ground_truth_numeric': job['gt'],
                    'correct': job['gt'] == status_numeric,
                })
                y_true.append(job['gt'])
                y_pred.append(status_numeric)
            scaffold_results.append(result_entry)
        else:
            annotated = draw_detections(orig, detections, args.clss_dict)

        if job['subdir']:
            out_path = os.path.join(args.output_dir, job['subdir'], os.path.basename(image_file))
        else:
            out_path = os.path.join(args.output_dir, os.path.basename(image_file))
        cv2.imwrite(out_path, annotated)
        saved_count += 1

    if args.scaffold_classification and scaffold_results:
        results_path = os.path.join(args.output_dir, 'scaffold_results.json')
        with open(results_path, 'w') as f:
            json.dump(scaffold_results, f, indent=2)
        print(f"  Scaffold results: {results_path}")
        print(f"  Predicted safe: {pred_safe_count}, Predicted unsafe: {pred_unsafe_count}")

    if args.eval_mode and y_true:
        metrics = compute_classification_metrics(y_true, y_pred)
        metrics_path = os.path.join(args.output_dir, 'classification_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print_classification_metrics(metrics)
        print(f"  Classification metrics: {metrics_path}")

    total_time = time.time() - start_time
    avg_fps = saved_count / total_inference_time if total_inference_time > 0 else 0

    print(f"\nInference complete:")
    print(f"  Images saved: {saved_count}/{num_images}")
    print(f"  Total wall time: {total_time:.2f} seconds")
    print(f"  Total inference time: {total_inference_time:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Output directory: {args.output_dir}")


def main():
    args = parse_arguments()

    if not boot(args):
        exit(-1)

    if not save_model(args):
        print("save_model fail")
        print("Error :", rt_core.get_last_error_code())
        exit(-1)
    print("Model saved successfully")

    if not make_instance(args):
        exit(-1)
    print("Instance created successfully")

    if not connect_instance(args):
        exit(-1)
    print("Instance connected successfully")

    load_post_processor(args)
    print("Post-processor loaded successfully")

    run_inference(args)

    rt_core.deinit_instance(args.interface, args.instance_name)


if __name__ == '__main__':
    main()
