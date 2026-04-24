#!/usr/bin/env python
# coding: utf-8

"""
Post-Processing Pipeline Comparison
Compare the modified vs working pipeline to find inconsistencies
"""

import os, sys
import json

def compare_pipelines():
    """Compare the two post-processing pipelines"""
    
    print("=== Post-Processing Pipeline Comparison ===")
    
    # Define the paths and configurations
    modified_config = {
        'file': 'ppe_example_started_org_modify.py',
        'model_name': 'BSNet0-20240820_0-YOLOv9',
        'model_path': '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/sep_23_model_416x416x3_inv-f.tachyrt',
        'config_path': '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json',
        'class_path': '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr/class.json',
        'post_process_module_path': '/home/dpi/raspberrypi_20241016/inference/example/utils/object_detection_yolov9/req_files_ppr'
    }
    
    working_config = {
        'file': 'object_detection_pic.py',
        'model_name': 'object_detection_yolov9',
        'model_path_template': '../utils/object_detection_yolov9/{}/model_{}_inv-f.tachyrt',
        'config_path_template': '../utils/object_detection_yolov9/{}/post_process_{}.json',
        'class_path_template': '../utils/object_detection_yolov9/{}/class.json',
        'post_process_module_path_template': '../utils/object_detection_yolov9/{}'
    }
    
    print("\n1. MODEL CONFIGURATION COMPARISON:")
    print("=" * 50)
    
    print(f"Modified file ({modified_config['file']}):")
    print(f"  Model name: {modified_config['model_name']}")
    print(f"  Model path: {modified_config['model_path']}")
    print(f"  Config path: {modified_config['config_path']}")
    print(f"  Class path: {modified_config['class_path']}")
    print(f"  Post-process module: {modified_config['post_process_module_path']}")
    
    print(f"\nWorking file ({working_config['file']}):")
    print(f"  Model name: {working_config['model_name']}")
    print(f"  Model path template: {working_config['model_path_template']}")
    print(f"  Config path template: {working_config['config_path_template']}")
    print(f"  Class path template: {working_config['class_path_template']}")
    print(f"  Post-process module template: {working_config['post_process_module_path_template']}")
    
    # Check if files exist
    print(f"\n2. FILE EXISTENCE CHECK:")
    print("=" * 50)
    
    files_to_check = [
        ('Model file', modified_config['model_path']),
        ('Config file', modified_config['config_path']),
        ('Class file', modified_config['class_path']),
        ('Post-process module', os.path.join(modified_config['post_process_module_path'], 'post_process.py'))
    ]
    
    for file_type, file_path in files_to_check:
        exists = os.path.exists(file_path)
        print(f"  {file_type}: {file_path}")
        print(f"    Exists: {exists}")
        if exists:
            size = os.path.getsize(file_path)
            print(f"    Size: {size} bytes")
        print()
    
    # Load and compare configurations
    print(f"\n3. CONFIGURATION COMPARISON:")
    print("=" * 50)
    
    try:
        modified_config_data = json.load(open(modified_config['config_path']))
        print("Modified config:")
        print(json.dumps(modified_config_data, indent=2))
    except Exception as e:
        print(f"Failed to load modified config: {e}")
    
    # Check what the working file would load
    print(f"\n4. WORKING FILE PATH RESOLUTION:")
    print("=" * 50)
    
    # Simulate what the working file would load with BSNet0-20240820_0-YOLOv9
    model_name = "BSNet0-20240820_0-YOLOv9"
    input_shape = "320x416x3"
    
    working_model_path = working_config['model_path_template'].format(model_name, input_shape)
    working_config_path = working_config['config_path_template'].format(model_name, input_shape)
    working_class_path = working_config['class_path_template'].format(model_name)
    working_module_path = working_config['post_process_module_path_template'].format(model_name)
    
    print(f"Working file would load:")
    print(f"  Model path: {working_model_path}")
    print(f"  Config path: {working_config_path}")
    print(f"  Class path: {working_class_path}")
    print(f"  Module path: {working_module_path}")
    
    # Check if these files exist
    working_files = [
        ('Model file', working_model_path),
        ('Config file', working_config_path),
        ('Class file', working_class_path),
        ('Post-process module', os.path.join(working_module_path, 'post_process.py'))
    ]
    
    print(f"\nWorking file paths existence:")
    for file_type, file_path in files_to_check:
        exists = os.path.exists(file_path)
        print(f"  {file_type}: {file_path}")
        print(f"    Exists: {exists}")
        print()
    
    # Load working config if it exists
    try:
        if os.path.exists(working_config_path):
            working_config_data = json.load(open(working_config_path))
            print("Working config:")
            print(json.dumps(working_config_data, indent=2))
        else:
            print("Working config file does not exist!")
    except Exception as e:
        print(f"Failed to load working config: {e}")
    
    print(f"\n5. KEY DIFFERENCES IDENTIFIED:")
    print("=" * 50)
    
    differences = []
    
    # Check model paths
    if modified_config['model_path'] != working_model_path:
        differences.append(f"Model path mismatch: Modified uses hardcoded path, Working uses template")
    
    # Check config paths
    if modified_config['config_path'] != working_config_path:
        differences.append(f"Config path mismatch: Modified uses hardcoded path, Working uses template")
    
    # Check class paths
    if modified_config['class_path'] != working_class_path:
        differences.append(f"Class path mismatch: Modified uses hardcoded path, Working uses template")
    
    # Check module paths
    if modified_config['post_process_module_path'] != working_module_path:
        differences.append(f"Module path mismatch: Modified uses hardcoded path, Working uses template")
    
    # Check model names
    if modified_config['model_name'] != working_config['model_name']:
        differences.append(f"Model name mismatch: Modified='{modified_config['model_name']}', Working='{working_config['model_name']}'")
    
    if differences:
        for i, diff in enumerate(differences, 1):
            print(f"  {i}. {diff}")
    else:
        print("  No differences found!")
    
    print(f"\n6. POTENTIAL ISSUES:")
    print("=" * 50)
    
    # Check if modified file is using wrong post-process module
    modified_module_path = modified_config['post_process_module_path']
    working_module_path = working_module_path
    
    if modified_module_path != working_module_path:
        print(f"⚠️  CRITICAL ISSUE: Modified file uses post-process module from:")
        print(f"   {modified_module_path}")
        print(f"   But should use:")
        print(f"   {working_module_path}")
        print(f"   This could cause different post-processing behavior!")
    
    # Check if configs are different
    try:
        if os.path.exists(working_config_path):
            working_config_data = json.load(open(working_config_path))
            if modified_config_data != working_config_data:
                print(f"⚠️  CONFIG DIFFERENCE: Modified and working configs are different!")
                print(f"   This could cause different post-processing behavior!")
    except:
        pass

if __name__ == '__main__':
    compare_pipelines()
