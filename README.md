this is running perfectly 

python object_detection_pic_322.py     --input_shape 416x416x3     --input_dir ../PPE_Ladder_hat/nipa_examples/

1_Mar_18_14_model_416x416x3_inv-f.tachyrt


to run iou

python3 evaluate_model_map_iou.py   --model "/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt"   --input_shape 416x416x3   --test_dir "/home/dpi/Desktop/inference__nov_migration/example/NIPA_complete/"   --post_process_module "/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process.py"


