#!/bin/bash

python object_detection_pic.py \
    --model BSNet0-20240820_0-YOLOv9 \
    --input_shape 320x416x3 \
    --input_dir ./image
