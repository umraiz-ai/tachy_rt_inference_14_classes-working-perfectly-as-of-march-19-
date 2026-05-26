export TACHY_INTERFACE=spi:host
#!/bin/bash

python object_detection_pic_322.py \
    --model BSNet0-20240820_0-YOLOv9 \
    --input_shape 320x416x3 \
    --input_dir ./image

export TACHY_INTERFACE=spi:host
cd ~/Desktop/inference__nov_migration/example/object_detection_yolo_coco-80cls

# After reboot or any hang: boot once or run 
 
 npu_invoke_example.py
 then
python object_detection_pic_322.py \
  --input_shape 416x416x3 \
  --input_dir ./image \
  --upload_firmware true \
  --path_firmware ../firmware

to view the output on example image, keep this value in post_process_xxx.json file 

    "OBJ_THRESHOLD":0.25,
    "NMS_THRESHOLD":0.2,
