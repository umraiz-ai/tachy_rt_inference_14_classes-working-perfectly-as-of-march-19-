#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import tachy_rt.core.functions as rt_core

sys.path.append('./utils/common')
from functions import read_json

# Class labels mapping (based on PPE dataset)
# These are placeholder values, replace with actual class names from your dataset
CLASS_NAMES = {
    '0': 'Helmet',
    '1': 'Person',
    '2': 'Worker'  # Based on the label file we saw
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate mAP and IoU for object detection model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model path to inference')
    
    parser.add_argument('--model_name', type=str, default="object_detection_yolov9",
                        help='Model name')
    
    parser.add_argument('--input_shape', type=str, required=True,
                        help='Input shape (HxWxD)')
    
    parser.add_argument('--test_dir', type=str, 
                        default='/home/dpi/raspberrypi_20241209/inference/example/complete_test_set_ppe',
                        help='Directory containing the test set (images and labels folders)')
    
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for considering a detection as true positive')

    parser.add_argument('--upload_firmware', type=str, default='false',
                        help='Upload firmware when using spi interface (true/false)')

    parser.add_argument('--path_firmware', type=str, default='./firmware/tachy-shield',
                        help='Firmware directory path for tachy_rt 3.2.2 boot()')

    parser.add_argument('--post_process_module', type=str, default=None,
                        help='Explicit path to post_process.py (optional)')
    
    args = parser.parse_args()
    
    # Environment check
    ENVS = os.environ
    if not "TACHY_INTERFACE" in ENVS:
        print("Environment \"TACHY_INTERFACE\" is not set")
        exit()
    
    args.interface = ENVS["TACHY_INTERFACE"]
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))
    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    
    # Get class dictionary
    try:
        args.class_path = os.path.dirname(args.model) + '/class.json'
        if os.path.exists(args.class_path):
            args.clss_dict = read_json(args.class_path)
        else:
            # Use the default class dictionary
            args.clss_dict = CLASS_NAMES
    except Exception as e:
        print(f"Error loading class dictionary: {e}")
        args.clss_dict = CLASS_NAMES
    
    return args

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

    # Upload firmware to device (tachy_rt 3.2.2 API)
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
    # Upload model
    ret = rt_core.save_model(args.interface, args.model_name, rt_core.MODEL_STORAGE_MEMORY, args.model, overwrite=True)
    return ret

def make_instance(args):
    # Clean stale instance (ignore errors)
    args.instance_name = f"{args.model_name}_inst"
    try:
        rt_core.deinit_instance(args.interface, args.instance_name)
    except Exception:
        pass
    try:
        rt_core.deinit_instance(args.interface, args.model_name)
    except Exception:
        pass

    # Make runtime config
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

    # Make engine (try both tokens used by different package variants)
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
    # Connect instance
    ret, args.instance = rt_core.connect_instance(args.interface, args.instance_name)
    if not ret:
        print("Connect instance fail")
        print("Error :", rt_core.get_last_error_code())
        return False
    return ret

def load_post_processor(args):
    # Prefer post-process artifacts near the model file, fallback to utils glob.
    model_dir = os.path.dirname(os.path.abspath(args.model))
    shape_cfg = f"post_process_{args.h}x{args.w}x3.json"
    post_config_paths = []
    if os.path.isdir(model_dir):
        post_config_paths.extend(glob.glob(os.path.join(model_dir, shape_cfg)))
        post_config_paths.extend(glob.glob(os.path.join(model_dir, "post_process*.json")))

    if not post_config_paths:
        post_config_paths = glob.glob('./utils/**/post_process*.json', recursive=True)

    if not post_config_paths:
        print("Cannot find post-processing config. Please specify the correct path.")
        exit(-1)

    args.post_config = read_json(post_config_paths[0])
    print(f"Using post-processing config: {post_config_paths[0]}")

    # Find post-processing module:
    # 1) explicit CLI path (if provided), 2) near config, 3) fallback glob.
    post_module_paths = []
    if args.post_process_module:
        if os.path.isfile(args.post_process_module):
            post_module_paths = [args.post_process_module]
        else:
            print(f"Provided --post_process_module not found: {args.post_process_module}")
            exit(-1)
    else:
        cfg_dir = os.path.dirname(post_config_paths[0])
        post_module_paths = glob.glob(os.path.join(cfg_dir, 'post_process.py'))
        if not post_module_paths:
            post_module_paths = glob.glob('./utils/**/post_process.py', recursive=True)

    if not post_module_paths:
        print("Cannot find post-processing module. Please specify the correct path.")
        exit(-1)

    post_module_dir = os.path.dirname(post_module_paths[0])
    sys.path.append(post_module_dir)
    args.post = __import__('post_process').Decoder(args.post_config)
    print(f"Using post-processing module: {post_module_paths[0]}")

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Get coordinates of intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate area of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate area of union
    area_union = area_box1 + area_box2 - area_inter
    
    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0
    
    return iou

def evaluate(args):
    # Get all test images
    image_files = sorted(glob.glob(os.path.join(args.test_dir, 'images', '*.jpg')))
    num_images = len(image_files)
    
    if num_images == 0:
        print(f"No images found in {args.test_dir}/images")
        return
    
    print(f"Evaluating {num_images} images...")
    
    all_detections = []  # Will hold all detections
    all_ground_truths = []  # Will hold all ground truths
    
    # Process all images
    start_time = time.time()
    total_inference_time = 0
    
    for image_file in tqdm(image_files):
        # Get image name without extension
        image_name = os.path.basename(image_file).rsplit('.', 1)[0]
        
        # Load corresponding label file
        label_file = os.path.join(args.test_dir, 'labels', os.path.basename(image_file).replace('.jpg', '.txt'))
        
        # Read ground truth boxes
        gt_boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:  # class, x_center, y_center, width, height
                        class_id = int(data[0])
                        # Convert YOLO format (center x,y,width,height normalized) to absolute coordinates
                        x_center = float(data[1]) * args.w
                        y_center = float(data[2]) * args.h
                        width = float(data[3]) * args.w
                        height = float(data[4]) * args.h
                        
                        # Calculate absolute coordinates [x1, y1, x2, y2]
                        x1 = max(0, x_center - (width / 2))
                        y1 = max(0, y_center - (height / 2))
                        x2 = min(args.w, x_center + (width / 2))
                        y2 = min(args.h, y_center + (height / 2))
                        
                        gt_boxes.append({
                            'class_id': class_id,
                            'box': [x1, y1, x2, y2],
                            'used': False  # To track matches during evaluation
                        })
        
        all_ground_truths.append(gt_boxes)
        
        # Load and prepare image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not read image: {image_file}")
            all_detections.append([])  # Add empty detections for this image
            continue
        
        # Resize image to model input size
        img = cv2.resize(img, (args.w, args.h))
        image = img.reshape(-1, args.h, args.w, 3)
        
        # Perform inference
        inference_start = time.time()
        args.instance.process([[image]])
        args.ret = args.instance.get_result()
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # Run post-processing
        detected_boxes = args.post.main(args.ret['buf'].view(np.float32), np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))
        
        # Convert detected boxes to our format
        detections = []
        for box in detected_boxes:
            if len(box) >= 6:
                confidence = float(box[0])
                class_id = int(box[1])
                x1, y1, x2, y2 = map(float, box[2:6])
                
                if confidence >= args.conf_threshold:
                    detections.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })
        
        all_detections.append(detections)
    
    total_time = time.time() - start_time
    avg_fps = num_images / total_inference_time if total_inference_time > 0 else 0
    
    # Calculate mAP
    print("\nCalculating mAP...")
    
    # Get all classes in the dataset
    all_classes = set()
    for gt_boxes in all_ground_truths:
        for box in gt_boxes:
            all_classes.add(box['class_id'])
    
    # Initialize AP storage
    ap_per_class = {}
    all_ious = []
    
    # For each class
    for class_id in all_classes:
        # Extract all detections and ground truths for this class
        class_detections = []
        class_ground_truths = []
        
        for i in range(len(image_files)):
            # Get detections for this class
            for det in all_detections[i]:
                if det['class_id'] == class_id:
                    class_detections.append({
                        'image_id': i,
                        'confidence': det['confidence'],
                        'box': det['box']
                    })
            
            # Get ground truths for this class
            gt_count = 0
            for gt in all_ground_truths[i]:
                if gt['class_id'] == class_id:
                    class_ground_truths.append({
                        'image_id': i,
                        'box': gt['box'],
                        'used': False
                    })
                    gt_count += 1
        
        # Sort detections by confidence score (highest first)
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Initialize true positives and false positives arrays
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        # Total number of ground truth objects for this class
        total_gt = len(class_ground_truths)
        
        # Process each detection
        for i, detection in enumerate(class_detections):
            img_id = detection['image_id']
            
            # Get ground truths for this image
            img_ground_truths = [gt for gt in class_ground_truths if gt['image_id'] == img_id]
            
            # Find best matching ground truth box
            best_iou = -1
            best_gt_idx = -1
            
            for j, gt in enumerate(img_ground_truths):
                if gt['used']:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(detection['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if detection is a true positive
            if best_gt_idx >= 0 and best_iou >= args.iou_threshold:
                if not img_ground_truths[best_gt_idx]['used']:
                    # Mark as used to prevent multiple detections
                    img_ground_truths[best_gt_idx]['used'] = True
                    tp[i] = 1  # Correct detection
                    all_ious.append(best_iou)
                else:
                    fp[i] = 1  # Multiple detection
            else:
                fp[i] = 1  # False detection
        
        # Compute precision/recall
        cumsum_fp = np.cumsum(fp)
        cumsum_tp = np.cumsum(tp)
        
        recalls = cumsum_tp / total_gt if total_gt > 0 else np.zeros_like(cumsum_tp)
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # Append sentinel values at start and end
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))
        
        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        
        # Find points where recall changes
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i)
        
        # Compute AP as area under curve
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i-1]) * mpre[i])
        
        # Store AP for this class
        class_name = args.clss_dict.get(str(class_id), f"Class_{class_id}")
        ap_per_class[class_name] = ap
    
    # Calculate mean AP across all classes
    mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0
    mean_iou = np.mean(all_ious) if all_ious else 0
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Total images processed: {num_images}")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"mAP@{args.iou_threshold}: {mean_ap:.4f}")
    print(f"Average IoU: {mean_iou:.4f}")
    print("\nAP per class:")
    
    for class_name, ap in ap_per_class.items():
        print(f"  - {class_name}: {ap:.4f}")
    
    # Create precision-recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(mrec[:-1], mpre[:-1], 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (mAP: {mean_ap:.4f})')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    
    # Create histogram of IoUs
    plt.figure(figsize=(10, 7))
    plt.hist(all_ious, bins=20, alpha=0.7)
    plt.axvline(x=args.iou_threshold, color='r', linestyle='--', label=f'Threshold ({args.iou_threshold})')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f'IoU Distribution (Mean IoU: {mean_iou:.4f})')
    plt.grid(True)
    plt.legend()
    plt.savefig('iou_distribution.png')
    
    # Save results to file
    results = {
        'mAP': float(mean_ap),
        'mean_IoU': float(mean_iou),
        'ap_per_class': {k: float(v) for k, v in ap_per_class.items()},
        'iou_threshold': args.iou_threshold,
        'conf_threshold': args.conf_threshold,
        'fps': float(avg_fps)
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to 'evaluation_results.json'")
    print("Precision-recall curve saved to 'precision_recall_curve.png'")
    print("IoU distribution saved to 'iou_distribution.png'")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Boot device (if needed)
    if not boot(args):
        exit(-1)
    
    # Save model to device
    if not save_model(args):
        print("save_model fail")
        print("Error :", rt_core.get_last_error_code())
        exit(-1)
    print("Model saved successfully")
    
    # Make instance for inference
    if not make_instance(args):
        exit(-1)
    print("Instance created successfully")
    
    # Connect to instance for inference
    if not connect_instance(args):
        exit(-1)
    print("Instance connected successfully")
    
    # Load post-processor
    load_post_processor(args)
    print("Post-processor loaded successfully")
    
    # Evaluate the model
    evaluate(args)
    
    # Cleanup
    rt_core.deinit_instance(args.interface, args.instance_name)

if __name__ == '__main__':
    main()
