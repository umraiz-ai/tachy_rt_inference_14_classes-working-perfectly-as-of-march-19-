this is running perfectly 
export TACHY_INTERFACE=spi:host

Do not forgtet to change the classes in posrt_process.py in this location

def split_logits(self, x, n, n_channels=(4, 4)):  # 4, 4 -> 4, 80

python object_detection_pic_322.py     --input_shape 416x416x3     --input_dir ../PPE_Ladder_hat/nipa_examples/

1_Mar_18_14_model_416x416x3_inv-f.tachyrt


to run iou

python3 evaluate_model_map_iou.py   --model "/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt"   --input_shape 416x416x3   --test_dir "/home/dpi/Desktop/inference__nov_migration/example/NIPA_complete/"   --post_process_module "/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process.py"


My npu got stuck 
first , flushed out the npu state with 
python3 flush_npu_state.py

then invoked npu

python3 npu_invoke_example.py   --model "/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt"   --input_shape 416x416x3   --upload_firmware true   --path_firmware "/home/dpi/Desktop/inference__nov_migration/example/firmware/"


to run evaluation on a nw type of nms

python npu_evaluate_map_iou.py \ 
    --model ./utils/object_detection_yolov9/req_files_ppr/mar_30_4_cls_model_416x416x3_inv-f.tachyrt \ 
    --post_process_config ./utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json \
    --post_process_module ./utils/object_detection_yolov9/req_files_ppr/post_process_per_class_nms.py \
    --class_json ./utils/object_detection_yolov9/req_files_ppr/class.json \
    --test_dir ./complete_test_set_ppe \
    --input_shape 416x416x3 \
    --iou_thresholds 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95


