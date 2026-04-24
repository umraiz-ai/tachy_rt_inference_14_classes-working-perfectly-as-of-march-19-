## How to run example script
### 1. Object Detection Yolo COCO 80cls
1-1. Run example script for picture <br>
``` bash
./bin/picture.sh
```
![image](./result_picture.png)

1-2. Run example script for sensor <br>
``` bash
./bin/sensor.sh
```
![image](./result_sensor.png)

1-3. How to add input image for picture example <br>

```
mv <image_path>.<format> ./image/input_<N>.<format>
```
This is updated file in sensor according to the API 3.2.2

python object_detection_sen_3.2.2.py     --model BSNet0-20240820_0-YOLOv9     --input_shape 320x416x3     --tx 2     --inverse_data  false     --inverse_sync  false     --inverse_clock false

Now I can use the the model, file and class.json and post_processxxx.json file explictly with 
python object_detection_sen_3.2.2.py \
  --model BSNet0-20240820_0-YOLOv9 \
  --input_shape 320x416x3 \
  --tx 2 \
  --inverse_data false \
  --inverse_sync false \
  --inverse_clock false \
  --model_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/oct_1_model_256x416x3_inv-f.tachyrt \
  --class_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/class.json \
  --post_config_path /home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process_256x416x3.json


  