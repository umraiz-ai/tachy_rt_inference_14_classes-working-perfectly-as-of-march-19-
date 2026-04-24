#!/usr/bin/env python
# coding: utf-8

"""
Post-Processing Debug Script
Analyze why post-processing produces identical bounding boxes
"""

import os, sys
sys.path.append('../utils/common')
import cv2
import glob
import numpy as np
import importlib
import json
import tachy_rt.core.functions as rt_core
from functions import *

def debug_post_processing():
    """Debug the post-processing step to find why boxes are identical"""
    
    # Setup model
    model_name = "post_debug"
    model_path = '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/sep_23_model_416x416x3_inv-f.tachyrt'
    config_path = '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json'
    class_path = '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/class.json'
    
    # Load configuration
    config = read_json(config_path)
    class_dict = read_json(class_path)
    
    print("=== Post-Processing Debug ===")
    print(f"Config: {config}")
    
    # Save model to NPU
    ret = rt_core.save_model('spi:host', model_name, rt_core.MODEL_STORAGE_MEMORY, model_path, overwrite=True)
    if not ret:
        print(f"Failed to save model: {rt_core.get_last_error_code()}")
        return
        
    # Create runtime config
    runtime_config = {
        "global": {
            "name": model_name,
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
    
    # Make engine
    ret = rt_core.make_instance('spi:host', model_name, model_name, "frame_split", runtime_config)
    if not ret:
        print(f"Failed to make instance: {rt_core.get_last_error_code()}")
        return
        
    # Connect instance
    ret, instance = rt_core.connect_instance('spi:host', model_name)
    if not ret:
        print(f"Failed to connect instance: {rt_core.get_last_error_code()}")
        return
        
    # Load post-processing module
    sys.path.append(os.path.dirname(config_path))
    post_processor = importlib.import_module('post_process').Decoder(config)
    
    # Test with first image
    image_file = './image/input_1.jpg'
    print(f"\nTesting with: {image_file}")
    
    # Load and preprocess image
    img = cv2.imread(image_file)
    img_resized = cv2.resize(img, (416, 416))
    image = img_resized.reshape(-1, 416, 416, 3)
    
    # Run inference
    instance.process([[image]])
    result = instance.get_result()
    raw_output = result['buf'].view(np.float32)
    
    print(f"Raw output shape: {raw_output.shape}")
    print(f"Raw output stats: min={raw_output.min():.3f}, max={raw_output.max():.3f}, mean={raw_output.mean():.3f}")
    
    # Debug post-processing step by step
    print(f"\n=== Post-Processing Step-by-Step ===")
    
    # Calculate grid points
    output_shapes = config['SHAPES_OUTPUT']
    total_grid = 0
    for shape in output_shapes:
        h, w, channels = shape
        grid_points = h * w
        total_grid += grid_points
        print(f"Scale {h}x{w}: {grid_points} grid points")
    
    print(f"Total grid points: {total_grid}")
    print(f"Expected output size: {total_grid * 8}")
    print(f"Actual output size: {len(raw_output)}")
    
    # Split logits manually
    n_channels = (4, 4)  # 4 box + 4 class
    box_elements = total_grid * n_channels[0]
    cls_elements = total_grid * n_channels[1]
    
    print(f"\nSplitting logits:")
    print(f"Box elements: {box_elements}")
    print(f"Class elements: {cls_elements}")
    print(f"Total expected: {box_elements + cls_elements}")
    
    if len(raw_output) >= box_elements + cls_elements:
        box_pred = np.reshape(raw_output[:box_elements], (-1, n_channels[0]))
        cls_pred = np.reshape(raw_output[box_elements:box_elements + cls_elements], (-1, n_channels[1]))
        
        print(f"Box predictions shape: {box_pred.shape}")
        print(f"Class predictions shape: {cls_pred.shape}")
        
        # Analyze box predictions
        print(f"\nBox predictions stats:")
        print(f"  Min: {box_pred.min():.3f}, Max: {box_pred.max():.3f}")
        print(f"  Mean: {box_pred.mean():.3f}, Std: {box_pred.std():.3f}")
        print(f"  Non-zero: {np.count_nonzero(box_pred)}")
        
        # Analyze class predictions
        print(f"\nClass predictions stats:")
        print(f"  Min: {cls_pred.min():.3f}, Max: {cls_pred.max():.3f}")
        print(f"  Mean: {cls_pred.mean():.3f}, Std: {cls_pred.std():.3f}")
        print(f"  Non-zero: {np.count_nonzero(cls_pred)}")
        
        # Check for identical predictions
        print(f"\nChecking for identical predictions:")
        unique_boxes = len(np.unique(box_pred, axis=0))
        unique_classes = len(np.unique(cls_pred, axis=0))
        print(f"  Unique box predictions: {unique_boxes}")
        print(f"  Unique class predictions: {unique_classes}")
        
        if unique_boxes == 1:
            print("  ⚠️  ALL BOX PREDICTIONS ARE IDENTICAL!")
        if unique_classes == 1:
            print("  ⚠️  ALL CLASS PREDICTIONS ARE IDENTICAL!")
            
        # Show some examples
        print(f"\nFirst 5 box predictions:")
        for i in range(min(5, len(box_pred))):
            print(f"  {i}: {box_pred[i]}")
            
        print(f"\nFirst 5 class predictions:")
        for i in range(min(5, len(cls_pred))):
            print(f"  {i}: {cls_pred[i]}")
    
    # Test with different threshold settings
    print(f"\n=== Testing Different Thresholds ===")
    
    # Original thresholds
    print(f"Original thresholds:")
    print(f"  OBJ_THRESHOLD: {config['OBJ_THRESHOLD']}")
    print(f"  NMS_THRESHOLD: {config['NMS_THRESHOLD']}")
    print(f"  PRE_THRESHOLD: {config['PRE_THRESHOLD']}")
    
    # Test with higher thresholds
    test_configs = [
        {'name': 'Higher OBJ', 'OBJ_THRESHOLD': 0.5, 'NMS_THRESHOLD': 0.3},
        {'name': 'Much Higher OBJ', 'OBJ_THRESHOLD': 0.8, 'NMS_THRESHOLD': 0.3},
        {'name': 'Original', 'OBJ_THRESHOLD': config['OBJ_THRESHOLD'], 'NMS_THRESHOLD': config['NMS_THRESHOLD']}
    ]
    
    for test_config in test_configs:
        print(f"\n--- {test_config['name']} ---")
        
        # Create modified config
        mod_config = config.copy()
        mod_config['OBJ_THRESHOLD'] = test_config['OBJ_THRESHOLD']
        mod_config['NMS_THRESHOLD'] = test_config['NMS_THRESHOLD']
        
        # Create new post-processor
        mod_post_processor = importlib.import_module('post_process').Decoder(mod_config)
        
        # Run post-processing
        reference = np.array([[0, 0, 415, 415]], dtype=np.float32)
        detections = mod_post_processor.main(raw_output, reference)
        
        print(f"  Detections: {len(detections)}")
        for i, detection in enumerate(detections):
            if len(detection) >= 6:
                prob, cls, x1, y1, x2, y2 = detection[:6]
                print(f"    {i+1}: class={int(cls)}, prob={prob:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Cleanup
    rt_core.deinit_instance('spi:host', model_name)
    print("\n=== Debug Complete ===")

if __name__ == '__main__':
    debug_post_processing()
