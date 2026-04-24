#!/bin/bash

# Test script for obj_det_modify.py
# This script runs the object detection with proper parameters

echo "Running Object Detection Test..."
echo "================================"

# Set the TACHY_INTERFACE environment variable if not set
if [ -z "$TACHY_INTERFACE" ]; then
    export TACHY_INTERFACE="spi:host"
    echo "Set TACHY_INTERFACE to: $TACHY_INTERFACE"
fi

# Run the object detection script
python3 obj_det_modify.py \
    --model "BSNet0-20240820_0-YOLOv9" \
    --input_shape "320x416x3" \
    --input_dir "./image" \
    --upload_firmware "true" \
    --path_firmware "../firmware/tachy-shield"

echo "================================"
echo "Test completed!"


