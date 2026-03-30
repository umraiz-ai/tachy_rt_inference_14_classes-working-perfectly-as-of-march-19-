#!/usr/bin/env python
# coding: utf-8
"""
VOC-style mAP@0.5 (simple)
python npu_evaluate_map_iou.py \
    --model ./utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt \
    --post_process_config ./utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json \
    --post_process_module ./utils/object_detection_yolov9/req_files_ppr/post_process.py \
    --class_json ./utils/object_detection_yolov9/req_files_ppr/class.json \
    --test_dir ./complete_test_set_ppe \
    --input_shape 416x416x3
"""




"""
NPU Object Detection Evaluation — mAP & IoU

Runs a .tachyrt YOLOv9 model on the NPU, compares predictions against
YOLO-format ground-truth labels and reports mAP@0.5, mAP@0.5:0.95,
per-class AP, and IoU statistics.

Example
-------
coco_style
python npu_evaluate_map_iou.py \
    --model ./utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt \
    --post_process_config ./utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json \
    --post_process_module ./utils/object_detection_yolov9/req_files_ppr/post_process.py \
    --class_json ./utils/object_detection_yolov9/req_files_ppr/class.json \
    --test_dir ./complete_test_set_ppe \
    --input_shape 416x416x3
"""

import os
import sys
import glob
import json
import time
import argparse
import importlib.util

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "utils", "common"))
from functions import read_json

import tachy_rt.core.functions as rt_core


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Evaluate a .tachyrt object-detection model (mAP / IoU)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--model", type=str, required=True,
                    help="Path to the .tachyrt model file")
    p.add_argument("--model_name", type=str, default="object_detection_yolov9",
                    help="Logical model name used by tachy_rt (default: object_detection_yolov9)")

    p.add_argument("--post_process_config", type=str, required=True,
                    help="Path to the post-process JSON config\n"
                         "e.g. post_process_416x416x3.json")
    p.add_argument("--post_process_module", type=str, required=True,
                    help="Path to post_process.py containing the Decoder class")

    p.add_argument("--class_json", type=str, required=True,
                    help="Path to class.json  (id → name mapping)")

    p.add_argument("--test_dir", type=str, required=True,
                    help="Root of test set.  Must contain images/ and labels/ sub-dirs.\n"
                         "Labels are YOLO format: class x_ctr y_ctr w h (normalised)")

    p.add_argument("--input_shape", type=str, required=True,
                    help="Model input shape as HxWxD  e.g. 416x416x3")

    p.add_argument("--iou_thresholds", type=str, default="0.5",
                    help="Comma-separated IoU thresholds for mAP.\n"
                         "Use '0.5' for VOC-style or\n"
                         "'0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95' for COCO-style")

    p.add_argument("--conf_threshold", type=float, default=None,
                    help="Extra confidence gate applied AFTER post-process.\n"
                         "Default: None (rely on post-process OBJ_THRESHOLD)")

    p.add_argument("--output_dir", type=str, default="./eval_output",
                    help="Directory to write result JSON, plots, etc.")

    p.add_argument("--upload_firmware", type=str, default="false")
    p.add_argument("--path_firmware", type=str, default="../firmware")

    p.add_argument("--save_visualisations", action="store_true",
                    help="Save per-image annotated PNGs (slow for large sets)")

    args = p.parse_args()

    if "TACHY_INTERFACE" not in os.environ:
        print('Environment variable "TACHY_INTERFACE" is not set')
        sys.exit(1)

    args.interface = os.environ["TACHY_INTERFACE"]
    dims = args.input_shape.split("x")
    args.h, args.w = int(dims[0]), int(dims[1])
    args.upload_firmware = args.upload_firmware.lower() == "true"
    args.iou_thresholds = [float(t) for t in args.iou_thresholds.split(",")]

    args.clss_dict = read_json(args.class_json)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


# ---------------------------------------------------------------------------
# NPU lifecycle  (boot → save_model → make_instance → connect)
# ---------------------------------------------------------------------------
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
        print("Missing firmware files:")
        for m in missing:
            print("  -", m)
        return None
    return {
        "spl":    {"path": spl,    "addr": "0x0"},
        "uboot":  {"path": uboot,  "addr": "0x2000_0000"},
        "kernel": {"path": kernel, "addr": "0x4000_0000"},
        "fpga":   {"path": fpga,   "addr": "0x3000_0000"},
    }


def boot(args):
    if "spi" not in args.interface or not args.upload_firmware:
        return True
    spi_type = args.interface.split(":")[-1]
    data = _build_boot_data(args.path_firmware)
    if data is None:
        return False
    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        print("Boot OK")
        return True
    print("Boot FAILED — error:", rt_core.get_last_error_code())
    return False


def save_model(args):
    ret = rt_core.save_model(
        args.interface, args.model_name,
        rt_core.MODEL_STORAGE_MEMORY, args.model, overwrite=True,
    )
    if not ret:
        print("save_model FAILED — error:", rt_core.get_last_error_code())
    return ret


def make_instance(args):
    args.instance_name = f"{args.model_name}_inst"
    for name in (args.instance_name, args.model_name):
        try:
            rt_core.deinit_instance(args.interface, name)
        except Exception:
            pass

    config = {
        "global": {
            "name": args.model_name,
            "data_type": rt_core.DTYPE_FLOAT16,
            "buf_num": 5,
            "max_batch": 1,
            "npu_mask": -1,
        },
        "input": [{
            "method": rt_core.INPUT_FMT_BINARY,
            "std": 255.0,
            "mean": 0.0,
        }],
        "output": {"reorder": True},
    }

    for algo in ("frame_spliter", "frame_splitter"):
        ret = rt_core.make_instance(
            args.interface, args.model_name,
            args.instance_name, algo, config,
        )
        if ret:
            print(f"make_instance OK  (algorithm={algo})")
            return True

    print("make_instance FAILED — error:", rt_core.get_last_error_code())
    return False


def connect_instance(args):
    ret, args.instance = rt_core.connect_instance(args.interface, args.instance_name)
    if not ret:
        print("connect_instance FAILED — error:", rt_core.get_last_error_code())
    return ret


# ---------------------------------------------------------------------------
# Post-processor loader  (imports post_process.py dynamically)
# ---------------------------------------------------------------------------
def load_post_processor(args):
    pp_config = read_json(args.post_process_config)
    print(f"Post-process config : {args.post_process_config}")

    spec = importlib.util.spec_from_file_location("post_process", args.post_process_module)
    pp_mod = importlib.util.module_from_spec(spec)

    pp_dir = os.path.dirname(os.path.abspath(args.post_process_module))
    if pp_dir not in sys.path:
        sys.path.insert(0, pp_dir)

    spec.loader.exec_module(pp_mod)
    args.post = pp_mod.Decoder(pp_config)
    print(f"Post-process module : {args.post_process_module}")
    print(f"  n_grid={args.post.n_grid}, n_cls={args.post.n_cls}, "
          f"obj_thr={float(args.post.obj_threshold)}, "
          f"nms_thr={float(args.post.iou_threshold)}")


# ---------------------------------------------------------------------------
# Ground-truth loading (YOLO txt → absolute coords at model input resolution)
# ---------------------------------------------------------------------------
def load_ground_truths(test_dir, img_w, img_h):
    """Return {image_stem: [{class_id, box:[x1,y1,x2,y2]}]} and image paths."""
    img_dir = os.path.join(test_dir, "images")
    lbl_dir = os.path.join(test_dir, "labels")

    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.jpeg"))
        + glob.glob(os.path.join(img_dir, "*.png"))
    )
    if not image_paths:
        print(f"No images found in {img_dir}")
        sys.exit(1)

    gt_by_image = {}
    valid_image_paths = []

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(lbl_dir, stem + ".txt")

        boxes = []
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                    x1 = max(0.0, xc - bw / 2.0)
                    y1 = max(0.0, yc - bh / 2.0)
                    x2 = min(float(img_w), xc + bw / 2.0)
                    y2 = min(float(img_h), yc + bh / 2.0)
                    boxes.append({"class_id": cls_id, "box": [x1, y1, x2, y2]})

        gt_by_image[stem] = boxes
        valid_image_paths.append(img_path)

    return gt_by_image, valid_image_paths


# ---------------------------------------------------------------------------
# IoU (single pair)
# ---------------------------------------------------------------------------
def compute_iou(box_a, box_b):
    """box format: [x1, y1, x2, y2]"""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# AP computation (all-point interpolation, same as COCO/VOC-2010+)
# ---------------------------------------------------------------------------
def compute_ap(recalls, precisions):
    """All-point interpolated AP."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap), mrec, mpre


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------
def evaluate(args):
    gt_by_image, image_paths = load_ground_truths(args.test_dir, args.w, args.h)
    n_images = len(image_paths)
    print(f"\nImages to evaluate : {n_images}")

    all_class_ids = set()
    for boxes in gt_by_image.values():
        for b in boxes:
            all_class_ids.add(b["class_id"])

    # --- Run inference on every image ----------------------------------
    # detections_by_image[stem] = [ {class_id, confidence, box} ]
    detections_by_image = {}
    all_matched_ious = []
    inference_times = []

    for idx, img_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] cannot read {img_path}")
            detections_by_image[stem] = []
            continue

        img_resized = cv2.resize(img, (args.w, args.h))
        blob = img_resized.reshape(-1, args.h, args.w, 3)

        t0 = time.time()
        args.instance.process([[blob]])
        raw = args.instance.get_result()
        dt = time.time() - t0
        inference_times.append(dt)

        buf = raw["buf"].view(np.float32)
        ref = np.array([[0, 0, args.w - 1, args.h - 1]], dtype=np.float32)
        anno = args.post.main(buf, ref)

        dets = []
        if anno is not None and len(anno) > 0:
            for row in anno:
                if len(row) < 6:
                    continue
                conf = float(row[0])
                cls_id = int(row[1])
                x1, y1, x2, y2 = map(float, row[2:6])
                if args.conf_threshold is not None and conf < args.conf_threshold:
                    continue
                dets.append({"class_id": cls_id, "confidence": conf,
                             "box": [x1, y1, x2, y2]})
                all_class_ids.add(cls_id)

        detections_by_image[stem] = dets

        if (idx + 1) % 50 == 0 or idx == n_images - 1:
            avg_ms = 1000.0 * np.mean(inference_times)
            print(f"  [{idx+1}/{n_images}]  avg inference {avg_ms:.1f} ms")

        # optional per-image visualisation
        if args.save_visualisations:
            _save_vis(args, img_resized, gt_by_image[stem], dets, stem)

    total_inf = sum(inference_times)
    avg_fps = n_images / total_inf if total_inf > 0 else 0.0

    # --- Per-threshold AP computation ----------------------------------
    stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    results_per_threshold = {}

    for iou_thr in args.iou_thresholds:
        ap_per_class = {}
        pr_curves = {}
        matched_ious_this_thr = []

        for cls_id in sorted(all_class_ids):
            cls_dets = []
            n_gt = 0
            gt_matched = defaultdict(lambda: defaultdict(bool))

            for s_idx, stem in enumerate(stems):
                for det in detections_by_image[stem]:
                    if det["class_id"] == cls_id:
                        cls_dets.append((s_idx, det["confidence"], det["box"]))
                for gt in gt_by_image[stem]:
                    if gt["class_id"] == cls_id:
                        n_gt += 1

            if n_gt == 0 and len(cls_dets) == 0:
                continue

            cls_dets.sort(key=lambda x: x[1], reverse=True)

            tp = np.zeros(len(cls_dets))
            fp = np.zeros(len(cls_dets))

            for d_idx, (s_idx, conf, det_box) in enumerate(cls_dets):
                stem = stems[s_idx]
                img_gts = [g for g in gt_by_image[stem] if g["class_id"] == cls_id]

                best_iou = 0.0
                best_gt_j = -1
                for j, gt in enumerate(img_gts):
                    iou = compute_iou(det_box, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_j = j

                if best_iou >= iou_thr and best_gt_j >= 0 and not gt_matched[stem][best_gt_j]:
                    tp[d_idx] = 1
                    gt_matched[stem][best_gt_j] = True
                    matched_ious_this_thr.append(best_iou)
                else:
                    fp[d_idx] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / n_gt if n_gt > 0 else np.zeros_like(cum_tp)
            precisions = cum_tp / (cum_tp + cum_fp)

            ap, mrec, mpre = compute_ap(recalls, precisions)

            cls_name = args.clss_dict.get(str(cls_id), f"class_{cls_id}")
            ap_per_class[cls_name] = {"ap": ap, "n_gt": n_gt,
                                      "n_det": len(cls_dets),
                                      "n_tp": int(cum_tp[-1]) if len(cum_tp) else 0}
            pr_curves[cls_name] = (mrec, mpre, ap)

        all_aps = [v["ap"] for v in ap_per_class.values()]
        mean_ap = float(np.mean(all_aps)) if all_aps else 0.0
        mean_iou = float(np.mean(matched_ious_this_thr)) if matched_ious_this_thr else 0.0

        results_per_threshold[iou_thr] = {
            "mAP": mean_ap,
            "mean_matched_IoU": mean_iou,
            "per_class": ap_per_class,
            "pr_curves": pr_curves,
            "matched_ious": matched_ious_this_thr,
        }

    # --- Aggregate mAP across all thresholds (COCO-style) -------------
    all_maps = [v["mAP"] for v in results_per_threshold.values()]
    overall_map = float(np.mean(all_maps)) if all_maps else 0.0

    # --- Print summary -------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Images evaluated     : {n_images}")
    print(f"  Total inference time : {total_inf:.2f} s")
    print(f"  Average FPS          : {avg_fps:.2f}")
    print(f"  IoU thresholds       : {args.iou_thresholds}")
    if len(args.iou_thresholds) > 1:
        print(f"  mAP@[{args.iou_thresholds[0]}:{args.iou_thresholds[-1]}] : {overall_map:.4f}")

    for iou_thr, res in sorted(results_per_threshold.items()):
        print(f"\n--- mAP @ IoU={iou_thr:.2f} : {res['mAP']:.4f}   "
              f"(mean matched IoU={res['mean_matched_IoU']:.4f}) ---")
        for cls_name, info in sorted(res["per_class"].items()):
            print(f"    {cls_name:20s}  AP={info['ap']:.4f}  "
                  f"(GT={info['n_gt']}, Det={info['n_det']}, TP={info['n_tp']})")
    print("=" * 70)

    # --- Save JSON results ---------------------------------------------
    json_out = {
        "overall_mAP": overall_map,
        "iou_thresholds": args.iou_thresholds,
        "n_images": n_images,
        "avg_fps": avg_fps,
        "model": os.path.abspath(args.model),
        "test_dir": os.path.abspath(args.test_dir),
        "input_shape": args.input_shape,
    }
    for iou_thr, res in results_per_threshold.items():
        key = f"mAP@{iou_thr}"
        json_out[key] = {
            "mAP": res["mAP"],
            "mean_matched_IoU": res["mean_matched_IoU"],
            "per_class": res["per_class"],
        }

    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults JSON → {json_path}")

    # --- Plots ---------------------------------------------------------
    _plot_pr_curves(args, results_per_threshold)
    _plot_iou_histogram(args, results_per_threshold)
    _plot_per_class_bar(args, results_per_threshold)

    return json_out


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
COLORS_GT  = (0, 255, 0)
COLORS_DET = (255, 0, 0)


def _save_vis(args, img, gt_boxes, det_boxes, stem):
    vis = img.copy()
    for g in gt_boxes:
        x1, y1, x2, y2 = map(int, g["box"])
        cls_name = args.clss_dict.get(str(g["class_id"]), str(g["class_id"]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS_GT, 2)
        cv2.putText(vis, f"GT:{cls_name}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_GT, 1)
    for d in det_boxes:
        x1, y1, x2, y2 = map(int, d["box"])
        cls_name = args.clss_dict.get(str(d["class_id"]), str(d["class_id"]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS_DET, 2)
        cv2.putText(vis, f"{cls_name} {d['confidence']:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_DET, 1)
    vis_dir = os.path.join(args.output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    cv2.imwrite(os.path.join(vis_dir, f"{stem}.png"), vis)


def _plot_pr_curves(args, results_per_threshold):
    for iou_thr, res in results_per_threshold.items():
        if not res["pr_curves"]:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        for cls_name, (mrec, mpre, ap) in sorted(res["pr_curves"].items()):
            ax.plot(mrec[:-1], mpre[:-1], label=f"{cls_name} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall  (mAP@{iou_thr:.2f} = {res['mAP']:.4f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        path = os.path.join(args.output_dir, f"pr_curve_iou{iou_thr:.2f}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"PR curve          → {path}")


def _plot_iou_histogram(args, results_per_threshold):
    for iou_thr, res in results_per_threshold.items():
        ious = res["matched_ious"]
        if not ious:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ious, bins=30, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(iou_thr, color="red", linestyle="--", linewidth=1.5,
                   label=f"threshold={iou_thr:.2f}")
        mean_iou = float(np.mean(ious))
        ax.axvline(mean_iou, color="orange", linestyle=":", linewidth=1.5,
                   label=f"mean={mean_iou:.3f}")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.set_title(f"IoU Distribution of True Positives  (threshold={iou_thr:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(args.output_dir, f"iou_hist_thr{iou_thr:.2f}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"IoU histogram     → {path}")


def _plot_per_class_bar(args, results_per_threshold):
    primary_thr = args.iou_thresholds[0]
    res = results_per_threshold[primary_thr]
    if not res["per_class"]:
        return

    names = sorted(res["per_class"].keys())
    aps = [res["per_class"][n]["ap"] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 5))
    bars = ax.bar(range(len(names)), aps, color="steelblue", edgecolor="black", linewidth=0.5)
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ap:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AP")
    ax.set_title(f"Per-Class AP  (mAP@{primary_thr:.2f} = {res['mAP']:.4f})")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(args.output_dir, "per_class_ap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Per-class AP bar  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()

    print("=" * 70)
    print("NPU Object-Detection Evaluation")
    print("=" * 70)
    print(f"  Model              : {args.model}")
    print(f"  Post-process cfg   : {args.post_process_config}")
    print(f"  Post-process module: {args.post_process_module}")
    print(f"  Class map          : {args.class_json}")
    print(f"  Test directory     : {args.test_dir}")
    print(f"  Input shape        : {args.h}x{args.w}x3")
    print(f"  IoU thresholds     : {args.iou_thresholds}")
    print(f"  Conf threshold     : {args.conf_threshold}")
    print(f"  Output directory   : {args.output_dir}")
    print()

    if not boot(args):
        sys.exit(1)

    if not save_model(args):
        sys.exit(1)
    print("Model uploaded to NPU")

    if not make_instance(args):
        sys.exit(1)

    if not connect_instance(args):
        sys.exit(1)
    print("Instance connected — ready for inference\n")

    load_post_processor(args)

    try:
        evaluate(args)
    finally:
        try:
            rt_core.deinit_instance(args.interface, args.instance_name)
            print("\nInstance cleaned up.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
