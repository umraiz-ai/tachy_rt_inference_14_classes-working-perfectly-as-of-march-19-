#!/usr/bin/env python
# coding: utf-8

"""
Run NPU object detection + optional scenario safe/unsafe classification.

Inference pipeline matches infernce-may-28/inf_end_to_end.py (letterbox, frame_split,
utils.yolov9 Decoder with built-in config).

Example (single folder):
  python classification_scenarios.py --model ./model.tachyrt --test_dir ./images --output_dir ./out

Example (scaffold safe/unsafe eval):
  python classification_scenarios.py \
      --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
      --safe_dir ./Scenarios/scaffold/safe \
      --unsafe_dir ./Scenarios/scaffold/unsafe \
      --output_dir ./Scenarios/scaffold/predictions \
      --scaffold_classification \
      --skip_hook_rule

Example (PPE safe/unsafe eval):
  python classification_scenarios.py \
      --model /path/to/your_ppe_model.tachyrt \
      --safe_dir /path/to/ppe/safe \
      --unsafe_dir /path/to/ppe/unsafe \
      --output_dir ./Scenarios/PPE/predictions \
      --ppe_classification \
      --ppe_conf_threshold 0.30

Example (worker counting eval):
  python classification_scenarios.py \
      --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
      --safe_dir ./Scenarios/Counting/safe/images \
      --unsafe_dir ./Scenarios/Counting/unsafe/images \
      --output_dir ./Scenarios/Counting/predictions \
      --counting \
      --counting_conf_threshold 0.30

Example (heavy machine safe/unsafe eval):
  python classification_scenarios.py \
      --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
      --safe_dir ./Scenarios/Heavy_Machine/safe \
      --unsafe_dir ./Scenarios/Heavy_Machine/unsafe \
      --output_dir ./Scenarios/Heavy_Machine/predictions \
      --heavy_machine_classification \
      --heavy_machine_conf_threshold 0.30

Example (lifted load safe/unsafe eval):
  python classification_scenarios.py \
      --model ./utils/object_detection_yolov9/req_files_ppr/may_20_cls_13_dpi_model_416x416x3_inv-f.tachyrt \
      --safe_dir ./Scenarios/Lifted_Load/safe \
      --unsafe_dir ./Scenarios/Lifted_Load/unsafe \
      --output_dir ./Scenarios/Lifted_Load/predictions \
      --lifted_load_classification \
      --lifted_load_conf_threshold 0.30
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

_MAY28_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'infernce-may-28')
)
_MAY28_UTILS = os.path.join(_MAY28_ROOT, 'utils')
for _p in (_MAY28_UTILS, _MAY28_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_scaffold_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'scaffold')
if _scaffold_dir not in sys.path:
    sys.path.append(_scaffold_dir)
from scaffold_classification import detect_scaffold, resolve_scaffold_class_ids

_ppe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'PPE')
if _ppe_dir not in sys.path:
    sys.path.append(_ppe_dir)
from Classification_PPE import detect_ppe, resolve_ppe_class_ids

_counting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'Counting')
if _counting_dir not in sys.path:
    sys.path.append(_counting_dir)
from Classification_Counting import (
    detect_worker_count,
    resolve_counting_class_ids,
    read_gt_count,
    draw_count_badge,
    compute_counting_metrics,
    print_counting_metrics,
)

_heavy_machine_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'Heavy_Machine'
)
if _heavy_machine_dir not in sys.path:
    sys.path.append(_heavy_machine_dir)
from Classification_Heavy_Machine import detect_heavy_machine, resolve_heavy_machine_class_ids

_lifted_load_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'Scenarios', 'Lifted_Load'
)
if _lifted_load_dir not in sys.path:
    sys.path.append(_lifted_load_dir)
from Classification_Lifted_Load import detect_lifted_load, resolve_lifted_load_class_ids

# Built-in defaults (same as infernce-may-28/inf_end_to_end.py inference())
DEFAULT_INPUT_H = 416
DEFAULT_INPUT_W = 416
DEFAULT_DECODER_CONFIG = {
    "SHAPES_INPUT": [416, 416, 3],
    "SHAPES_OUTPUT": [
        [52, 52, 17],
        [26, 26, 17],
        [13, 13, 17],
    ],
    "NMS_THRESHOLD": 0.2,
    "OBJ_THRESHOLD": 0.25,
    "N_CLASSES": 13,
    "N_MAX_OBJ": 100,
}

# 13-class label map for drawing / scaffold role resolution
DEFAULT_CLASS_DICT = {
    "0": "cement_truck",
    "1": "compactor",
    "2": "dump_truck",
    "3": "excavator",
    "4": "grader",
    "5": "mobile_crane",
    "6": "tower_crane",
    "7": "worker",
    "8": "Hardhat",
    "9": "Red_Hardhat",
    "10": "scaffolds",
    "11": "Lifted Load",
    "12": "Hook",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run object detection inference on a directory of images and save annotated outputs'
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the .tachyrt model file')

    parser.add_argument('--model_name', type=str, default="object_detection_yolov9",
                        help='Logical model name used by tachy_rt')

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
        help='Confidence gate for detections used in scaffold rules (default: 0.30)',
    )

    parser.add_argument(
        '--scaffold_min_conf',
        type=float,
        default=0.50,
        help='Min scaffold detection confidence to enable hook/vertical rules (default: 0.50)',
    )

    parser.add_argument(
        '--skip_hook_rule',
        action='store_true',
        help='Disable hook fastening rule (helmet + vertical rules still apply)',
    )

    parser.add_argument(
        '--ppe_classification',
        action='store_true',
        help='Apply PPE helmet safe/unsafe rules on NPU detections',
    )

    parser.add_argument(
        '--ppe_conf_threshold',
        type=float,
        default=0.30,
        help='Confidence gate for detections used in PPE rules (default: 0.30)',
    )

    parser.add_argument(
        '--counting',
        action='store_true',
        help='Count workers from NPU detections (eval uses label .txt files for GT counts)',
    )

    parser.add_argument(
        '--counting_conf_threshold',
        type=float,
        default=0.30,
        help='Confidence gate for worker detections used in counting (default: 0.30)',
    )

    parser.add_argument(
        '--safe_labels_dir',
        type=str,
        default=None,
        help='GT count labels for --safe_dir images (default: ../labels if safe_dir ends with /images)',
    )

    parser.add_argument(
        '--unsafe_labels_dir',
        type=str,
        default=None,
        help='GT count labels for --unsafe_dir images (default: ../labels if unsafe_dir ends with /images)',
    )

    parser.add_argument(
        '--heavy_machine_classification',
        action='store_true',
        help='Apply heavy machine safe/unsafe rules (helmet, signal man, proximity)',
    )

    parser.add_argument(
        '--heavy_machine_conf_threshold',
        type=float,
        default=0.30,
        help='Confidence gate for detections used in heavy machine rules (default: 0.30)',
    )

    parser.add_argument(
        '--heavy_machine_danger_dist',
        type=float,
        default=2.0,
        help='Proximity violation distance threshold in meters (default: 2.0)',
    )

    parser.add_argument(
        '--heavy_machine_no_default_homography',
        action='store_true',
        help='Disable default ground-plane homography (proximity check skipped without SEG tag)',
    )

    parser.add_argument(
        '--skip_signal_man_rule',
        action='store_true',
        help='Disable signal-man rule for heavy machine scenario',
    )

    parser.add_argument(
        '--lifted_load_classification',
        action='store_true',
        help='Apply lifted load safe/unsafe rules (red helmet, helmets, danger zone)',
    )

    parser.add_argument(
        '--lifted_load_conf_threshold',
        type=float,
        default=0.30,
        help='Confidence gate for detections used in lifted load rules (default: 0.30)',
    )

    parser.add_argument(
        '--skip_red_helmet_rule',
        action='store_true',
        help='Disable red-helmet (signaler) rule for lifted load scenario',
    )

    parser.add_argument(
        '--lifted_load_require_helmet_everywhere',
        action='store_true',
        help='Require helmets on all workers when lifted load is present (default: lift area only)',
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
    args.h = DEFAULT_INPUT_H
    args.w = DEFAULT_INPUT_W
    args.instance_name = args.model_name
    args.upload_firmware = args.upload_firmware.lower() == 'true'
    args.clss_dict = DEFAULT_CLASS_DICT.copy()
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

    scenario_flags = [
        args.scaffold_classification,
        args.ppe_classification,
        args.counting,
        args.heavy_machine_classification,
        args.lifted_load_classification,
    ]
    if sum(scenario_flags) > 1:
        print("Error: scenario flags (--scaffold_classification, --ppe_classification, "
              "--counting, --heavy_machine_classification, --lifted_load_classification) "
              "are mutually exclusive.")
        exit(1)

    if args.eval_mode and not any(scenario_flags):
        print("Error: a scenario flag is required with --safe_dir/--unsafe_dir "
              "(scaffold, ppe, counting, heavy_machine, or lifted_load_classification).")
        exit(1)

    if args.scaffold_classification:
        args.scaffold_class_ids = resolve_scaffold_class_ids(args.clss_dict)
        print(f"Scaffold class IDs: {args.scaffold_class_ids}")
        if args.skip_hook_rule:
            print("Hook rule: DISABLED (--skip_hook_rule)")

    if args.ppe_classification:
        args.ppe_class_ids = resolve_ppe_class_ids(args.clss_dict)
        print(f"PPE class IDs: {args.ppe_class_ids}")

    if args.heavy_machine_classification:
        args.heavy_machine_class_ids = resolve_heavy_machine_class_ids(args.clss_dict)
        print(f"Heavy machine class IDs: {args.heavy_machine_class_ids}")
        if args.heavy_machine_no_default_homography:
            print("Heavy machine homography: DISABLED (proximity may be skipped)")
        if args.skip_signal_man_rule:
            print("Signal-man rule: DISABLED (--skip_signal_man_rule)")

    if args.lifted_load_classification:
        args.lifted_load_class_ids = resolve_lifted_load_class_ids(args.clss_dict)
        print(f"Lifted load class IDs: {args.lifted_load_class_ids}")
        if args.skip_red_helmet_rule:
            print("Red-helmet rule: DISABLED (--skip_red_helmet_rule)")
        if args.lifted_load_require_helmet_everywhere:
            print("Lifted load helmets: required on all workers when load present")

    if args.counting:
        args.counting_class_ids = resolve_counting_class_ids(args.clss_dict)
        print(f"Counting class IDs: {args.counting_class_ids}")
        if args.eval_mode:
            args.safe_labels_dir = resolve_labels_dir(args.safe_dir, args.safe_labels_dir)
            args.unsafe_labels_dir = resolve_labels_dir(args.unsafe_dir, args.unsafe_labels_dir)
            if not args.safe_labels_dir or not os.path.isdir(args.safe_labels_dir):
                print(f"Error: safe labels directory not found (tried --safe_labels_dir or auto-detect near {args.safe_dir})")
                exit(1)
            if not args.unsafe_labels_dir or not os.path.isdir(args.unsafe_labels_dir):
                print(f"Error: unsafe labels directory not found (tried --unsafe_labels_dir or auto-detect near {args.unsafe_dir})")
                exit(1)
            print(f"Counting label dirs: safe={args.safe_labels_dir}, unsafe={args.unsafe_labels_dir}")

    return args


IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP', '*.webp', '*.WEBP')


def resolve_labels_dir(image_dir, explicit_labels_dir=None):
    """Resolve GT label directory for counting (sibling labels/ when image_dir is .../images)."""
    if explicit_labels_dir:
        return explicit_labels_dir
    normalized = image_dir.rstrip(os.sep)
    if os.path.basename(normalized) == 'images':
        return os.path.join(os.path.dirname(normalized), 'labels')
    candidate = os.path.join(normalized, 'labels')
    if os.path.isdir(candidate):
        return candidate
    return None


def collect_image_files(directory):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def build_counting_eval_jobs(args):
    jobs = []
    for split, img_dir, label_dir in (
        ('safe', args.safe_dir, args.safe_labels_dir),
        ('unsafe', args.unsafe_dir, args.unsafe_labels_dir),
    ):
        for path in collect_image_files(img_dir):
            basename = os.path.basename(path)
            try:
                gt_count = read_gt_count(label_dir, basename)
            except FileNotFoundError as e:
                print(f"[WARN] {e} -> skipping")
                continue
            jobs.append({
                'path': path,
                'subdir': split,
                'gt_label': split,
                'gt_count': gt_count,
            })
    return jobs


def build_image_jobs(args):
    if args.eval_mode and args.counting:
        return build_counting_eval_jobs(args)

    if args.eval_mode:
        jobs = []
        for path in collect_image_files(args.safe_dir):
            jobs.append({'path': path, 'gt': 1, 'subdir': 'safe', 'gt_label': 'safe'})
        for path in collect_image_files(args.unsafe_dir):
            jobs.append({'path': path, 'gt': 0, 'subdir': 'unsafe', 'gt_label': 'unsafe'})
        return jobs

    return [
        {'path': path, 'gt': None, 'subdir': '', 'gt_label': None, 'gt_count': None}
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


def normalize(image, mean, var):
    return (image - mean) / var


def letterbox_preprocess(bgr_image, rh, rw, mean=0.0, std=255.0):
    """Letterbox preprocess matching infernce-may-28/inf_end_to_end.py."""
    h, w, _ = bgr_image.shape
    image = bgr_image
    gain = min(rh / h, rw / w)
    gain = min(gain, 1.0)
    new_unpad = (int(round(w * gain)), int(round(h * gain)))
    pad_x = (rw - new_unpad[0]) / 2
    pad_y = (rh - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )[:, :, ::-1]
    npu_input = normalize(image.astype(np.float32), mean, std)[None, ...]

    ref_x1 = -pad_x / gain
    ref_y1 = -pad_y / gain
    ref_x2 = ref_x1 + (rw / gain) - 1
    ref_y2 = ref_y1 + (rh / gain) - 1
    ref = np.array([ref_x1, ref_y1, ref_x2, ref_y2], dtype=np.float32)[None, ...]

    return npu_input, ref


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
                "std": 1.0,
                "mean": 0.0,
                "tx": -1,
            }
        ],
        "output": {
            "reorder": True
        }
    }

    ret = rt_core.make_instance(
        args.interface,
        args.model_name,
        args.model_name,
        "frame_split",
        args.config,
    )
    if ret:
        print("make_instance success with algorithm: frame_split")
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
    from utils.yolov9 import Decoder
    args.post = Decoder(DEFAULT_DECODER_CONFIG)
    print(f"Using infernce-may-28 Decoder: {_MAY28_ROOT}/utils/yolov9.py")
    print(f"  input {DEFAULT_INPUT_H}x{DEFAULT_INPUT_W}, "
          f"obj_thr={DEFAULT_DECODER_CONFIG['OBJ_THRESHOLD']}, "
          f"nms_thr={DEFAULT_DECODER_CONFIG['NMS_THRESHOLD']}")


def npu_detect(args, orig):
    npu_input, ref = letterbox_preprocess(orig, args.h, args.w)

    args.instance.process([[npu_input]])
    logits = args.instance.get_result()['buf']
    detected_boxes = args.post.main(logits, ref)

    if detected_boxes is None or len(detected_boxes) == 0:
        return []

    orig_h, orig_w = orig.shape[:2]
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
                max(0, min(orig_w - 1, int(round(x1)))),
                max(0, min(orig_h - 1, int(round(y1)))),
                max(0, min(orig_w - 1, int(round(x2)))),
                max(0, min(orig_h - 1, int(round(y2)))),
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

    scenario = None
    if args.scaffold_classification:
        scenario = 'scaffold'
    elif args.ppe_classification:
        scenario = 'ppe'
    elif args.counting:
        scenario = 'counting'
    elif args.heavy_machine_classification:
        scenario = 'heavy_machine'
    elif args.lifted_load_classification:
        scenario = 'lifted_load'

    if args.eval_mode:
        os.makedirs(os.path.join(args.output_dir, 'safe'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'unsafe'), exist_ok=True)
        if scenario == 'counting':
            mode = "counting eval"
        else:
            mode = f"{scenario} eval (safe/unsafe)" if scenario else "eval (safe/unsafe)"
    else:
        if scenario == 'counting':
            mode = "counting"
        else:
            mode = f"{scenario} classification" if scenario else "detection"

    print(f"Processing {num_images} images ({mode})...")
    start_time = time.time()
    total_inference_time = 0
    saved_count = 0
    scenario_results = []
    pred_safe_count = 0
    pred_unsafe_count = 0
    total_skipped_workers = 0
    y_true = []
    y_pred = []
    gt_counts = []
    pred_counts = []

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
                scaffold_min_conf=args.scaffold_min_conf,
                skip_hook_rule=args.skip_hook_rule,
            )
        elif args.ppe_classification:
            annotated, status_numeric, reasons = detect_ppe(
                orig,
                detections,
                args.ppe_class_ids,
                conf_threshold=args.ppe_conf_threshold,
            )
        elif args.counting:
            annotated, worker_count, skipped_workers = detect_worker_count(
                orig,
                detections,
                args.counting_class_ids,
                conf_threshold=args.counting_conf_threshold,
            )
            total_skipped_workers += skipped_workers
            gt_count = job.get('gt_count')
            annotated = draw_count_badge(
                annotated,
                job['subdir'].upper() if job.get('subdir') else 'IMAGE',
                gt_count,
                worker_count,
            )
            status_numeric = None
            reasons = None
        elif args.heavy_machine_classification:
            annotated, status_numeric, reasons = detect_heavy_machine(
                orig,
                detections,
                args.heavy_machine_class_ids,
                conf_threshold=args.heavy_machine_conf_threshold,
                danger_dist_meters=args.heavy_machine_danger_dist,
                use_default_homography=not args.heavy_machine_no_default_homography,
                skip_signal_man_rule=args.skip_signal_man_rule,
            )
        elif args.lifted_load_classification:
            annotated, status_numeric, reasons = detect_lifted_load(
                orig,
                detections,
                args.lifted_load_class_ids,
                conf_threshold=args.lifted_load_conf_threshold,
                require_helmet_only_in_lift_area=not args.lifted_load_require_helmet_everywhere,
                skip_red_helmet_rule=args.skip_red_helmet_rule,
            )
        else:
            annotated = draw_detections(orig, detections, args.clss_dict)
            status_numeric = None
            reasons = None

        if scenario == 'counting':
            result_entry = {
                'file': os.path.basename(image_file),
                'pred_count': worker_count,
                'skipped_workers': skipped_workers,
            }
            if args.eval_mode:
                err = worker_count - job['gt_count']
                result_entry.update({
                    'subdir': job['subdir'],
                    'split': job['gt_label'],
                    'gt_count': job['gt_count'],
                    'error': err,
                    'correct': err == 0,
                })
                gt_counts.append(job['gt_count'])
                pred_counts.append(worker_count)
            scenario_results.append(result_entry)
        elif scenario:
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
            scenario_results.append(result_entry)

        if job['subdir']:
            out_path = os.path.join(args.output_dir, job['subdir'], os.path.basename(image_file))
        else:
            out_path = os.path.join(args.output_dir, os.path.basename(image_file))
        cv2.imwrite(out_path, annotated)
        saved_count += 1

    if scenario and scenario_results:
        results_name = scenario
        results_path = os.path.join(args.output_dir, f'{results_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(scenario_results, f, indent=2)
        print(f"  {scenario.capitalize()} results: {results_path}")
        if scenario == 'counting':
            print(f"  Total workers skipped (too small): {total_skipped_workers}")
        else:
            print(f"  Predicted safe: {pred_safe_count}, Predicted unsafe: {pred_unsafe_count}")

    if args.counting and args.eval_mode and gt_counts:
        metrics = compute_counting_metrics(gt_counts, pred_counts)
        metrics_path = os.path.join(args.output_dir, 'counting_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print_counting_metrics(metrics)
        print(f"  Counting metrics: {metrics_path}")
    elif args.eval_mode and y_true and not args.counting:
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
