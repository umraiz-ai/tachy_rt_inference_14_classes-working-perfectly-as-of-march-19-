#!/bin/bash

python object_detection_sen.py \
    --model BSNet0-20240820_0-YOLOv9 \
    --input_shape 320x416x3 \
    --tx 2 \
    --inverse_data  false \
    --inverse_sync  false \
    --inverse_clock false
