#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.append('../utils/common')
# sys.path.append('../utils/BSNet0-20240820_0-YOLOv9')
import cv2
import glob
import argparse
import numpy as np
import importlib
import time
import tachy_rt.core.functions as rt_core
from tachy_rt.utils.constants import *
from threading import Thread
from functions import *

def create_args():
    def get_parser():
        """
        Get the parser.
        :return: parser
        """
    
        parser = argparse.ArgumentParser(description='Deep learning example script')
    
        parser.add_argument('--model', type=str,
                            help='Model to inference',
                            required=True)

        parser.add_argument('--input_shape', type=str,
                            help='Target input shape(HxWxD)',
                            default=None)

        parser.add_argument('--input_dir', type=str,
                            help='Directory for inputs',
                            required=True)

        parser.add_argument('--video_input', type=str,
                    help='Path to input video file (optional). If provided, video will be processed instead of image directory',
                    default=None)

        parser.add_argument('--video_output', type=str,
                    help='Path for output video (when --video_input is used)',
                    default='result_video.mp4')

        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='true')

        parser.add_argument('--path_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='../firmware/tachy-shield')

    
        args = parser.parse_args()

        ENVS = os.environ
        if not "TACHY_INTERFACE" in ENVS:
            print("Environment \"TACHY_INTERFACE\" is not set")
            exit()

        args.interface = ENVS["TACHY_INTERFACE"]

        return args
    
    args = get_parser()

    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))
    args.model_name = "object_detection_yolov9"

    args.model_path = '/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/oct_1_model_256x416x3_inv-f.tachyrt'
    args.clss_dict  = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/class.json'.format(args.model))
    # Prepare image inputs unless a video input was provided
    if args.video_input is None:
        args.images_input = glob.glob(f'{args.input_dir}/input*')
        args.images_input.sort()
        args.images_input = [cv2.resize(cv2.imread(f), (args.w,args.h)) for f in args.images_input]
    else:
        args.images_input = []


    return args

def boot(args):
    if 'spi' not in args.interface:
        return

    spi_type = args.interface.split(":")[-1]

    ''' Check if uploading firmware or just initializing SPI '''
    if args.upload_firmware:
        data = {
            "spl": {"path": os.path.join(args.path_firmware, "image.ub"), "addr": "0x0"}
        }
    else:
        # Pass an empty dictionary just to initialize the FTDI connection
        data = {} 
    
    # Replace DEV_TACHY_BS with 0
    ret = rt_core.boot(spi_type, 0, data)
    
    if ret:
        print("Success to boot. Check the status via uart or other api")
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)


def save_model(args):
    ''' Upload model '''
    # Shifted to positional arguments to match the working example
    ret = rt_core.save_model(args.interface, args.model_name, rt_core.MODEL_STORAGE_MEMORY, args.model_path, overwrite=True)
    return ret

def make_instance(args):
    ''' Make runtime config '''
    args.config = \
    {
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
                "std": [255.0, 255.0, 255.0],
                "mean": [0.0, 0.0, 0.0]
            }
        ],
        "output": {
            "reorder": True
        }
    }

    ''' Make engine '''
    # Positional arguments and "frame_split" instead of "frame_spliter"
    ret = rt_core.make_instance(args.interface, args.model_name, args.model_name, "frame_split", args.config)

    return ret

def connect_instance(args):
    ''' Connect instance '''
    ret, args.instance = rt_core.connect_instance(args.interface, args.model_name)
    if not ret:
        print("connect instance fail")
        print("error :", rt_core.get_last_error_code())
        exit()

    return ret

def inference(args):
    ''' Load Input & Do inference'''
    # Load post processing once
    args.post_config = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process_256x416x3.json')
    sys.path.append('../utils/object_detection_yolov9/req_files_ppr')
    args.post = importlib.import_module('post_process').Decoder(args.post_config)

    # If a video input is provided, stream frames and write an output video side-by-side
    if args.video_input:
        cap = cv2.VideoCapture(args.video_input)
        if not cap.isOpened():
            print('Failed to open video:', args.video_input)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w, out_h = args.w * 2, args.h
        writer = cv2.VideoWriter(args.video_output, fourcc, fps, (out_w, out_h))

        frame_idx = 0
        total_processing_time = 0
        total_start_time = time.time()  # Time for the entire video processing
        
        # Tracking time metrics
        total_inference_time = 0
        total_postprocess_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()
            
            # resize to model input
            img = cv2.resize(frame, (args.w, args.h))
            image = img.reshape(-1, args.h, args.w, 3)
            
            # Measure precise model inference time
            inference_start = time.time()
            args.instance.process(data=image)
            args.ret = args.instance.get_result()
            inference_end = time.time()
            inference_time = inference_end - inference_start
            total_inference_time += inference_time
            
            # Measure post-processing time
            postprocess_start = time.time()
            args.anno = args.post.main(args.ret['buf'].view(np.float32), np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))
            postprocess_end = time.time()
            postprocess_time = postprocess_end - postprocess_start
            total_postprocess_time += postprocess_time

            annotated = img.copy()
            for box in args.anno:
                _cls = args.clss_dict[str(box[1].astype(np.int32))]
                x0,y0,x1,y1 = box[2:6].astype(np.int32)
                annotated = cv2.rectangle(annotated, (x0,y0), (x1,y1), (255,0,0), 3)
                annotated = cv2.putText(annotated, _cls, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Calculate frame processing time (includes everything)
            frame_time = time.time() - frame_start_time
            total_processing_time += frame_time
            
            combined = np.concatenate([img, annotated], axis=1)
            writer.write(combined)
            frame_idx += 1

        # Calculate overall FPS for the entire video
        total_time = time.time() - total_start_time
        overall_fps = frame_idx / total_time if total_time > 0 else 0
        
        # Calculate model inference and post-processing metrics
        inference_fps = frame_idx / total_inference_time if total_inference_time > 0 else 0
        postprocess_fps = frame_idx / total_postprocess_time if total_postprocess_time > 0 else 0
        inference_plus_postprocess_fps = frame_idx / (total_inference_time + total_postprocess_time) if (total_inference_time + total_postprocess_time) > 0 else 0
        
        cap.release()
        writer.release()
        
        print(f'Wrote annotated video to {args.video_output} ({frame_idx} frames)')
        print('-' * 50)
        print(f'MODEL INFERENCE ONLY:')
        print(f'  - Total time: {total_inference_time:.2f} seconds')
        print(f'  - FPS: {inference_fps:.2f}')
        print(f'INFERENCE + POST-PROCESSING:')
        print(f'  - Total time: {total_inference_time + total_postprocess_time:.2f} seconds')
        print(f'  - FPS: {inference_plus_postprocess_fps:.2f}')
        print(f'OVERALL (including I/O):')
        print(f'  - Total time: {total_time:.2f} seconds')
        print(f'  - FPS: {overall_fps:.2f}')
    else:
        args.predicts = []
        total_time = 0
        total_inference_time = 0
        total_postprocess_time = 0
        num_images = len(args.images_input)
        
        print(f"Processing {num_images} images...")
        
        for i, img in enumerate(args.images_input):
            start_time = time.time()
            
            image = img.reshape(-1, args.h, args.w, 3)
            
            # Measure model inference time
            inference_start = time.time()
            args.instance.process(data=image)
            args.ret = args.instance.get_result()
            inference_end = time.time()
            
            # Measure post-processing time
            postprocess_start = time.time()
            args.anno = args.post.main(args.ret['buf'].view(np.float32), np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))
            postprocess_end = time.time()
            post_process_time = postprocess_end - postprocess_start
            total_postprocess_time += post_process_time

            # Overall processing time (including all operations)
            overall_time = time.time() - start_time
            total_time += overall_time
            
            processed_img = img.copy()
            for box in args.anno:
                _cls = args.clss_dict[str(box[1].astype(np.int32))]
                x0,y0,x1,y1 = box[2:6].astype(np.int32)
                processed_img = cv2.rectangle(processed_img, (x0,y0), (x1,y1), (255,0,0), 3)
                processed_img = cv2.putText(processed_img, _cls, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            args.predicts.append(processed_img)
        
        # Calculate average FPS across all images
        inference_fps = num_images / total_inference_time if total_inference_time > 0 else 0
        inference_plus_postprocess_fps = num_images / (total_inference_time + total_postprocess_time) if (total_inference_time + total_postprocess_time) > 0 else 0
        overall_fps = num_images / total_time if total_time > 0 else 0
        
        print('-' * 50)
        print(f'MODEL INFERENCE ONLY:')
        print(f'  - Total time: {total_inference_time:.2f} seconds')
        print(f'  - FPS: {inference_fps:.2f}')
        print(f'INFERENCE + POST-PROCESSING:')
        print(f'  - Total time: {total_inference_time + total_postprocess_time:.2f} seconds')
        print(f'  - FPS: {inference_plus_postprocess_fps:.2f}')
        print(f'OVERALL (including I/O):')
        print(f'  - Total time: {total_time:.2f} seconds')
        print(f'  - FPS: {overall_fps:.2f}')

    rt_core.deinit_instance(itf=args.interface, instance=args.model_name)

def display(args):
    # If video input was used, the annotated video has already been written in inference().
    if getattr(args, 'video_input', None):
        print('Video mode: output video written by inference(); skipping display()')
        return

    # Otherwise, create a single side-by-side image from the processed images
    if not getattr(args, 'images_input', None) or not getattr(args, 'predicts', None):
        print('No image results to display')
        return

    inputs    = np.concatenate(args.images_input, axis=0)
    total_img = inputs
    predicts = np.concatenate(args.predicts, axis=0)
    total_img = np.concatenate([total_img, predicts], axis=1)

    #cv2.imshow('', total_img)
    #cv2.waitKey(0)
    cv2.imwrite('result.png', total_img)

if __name__ == '__main__':
    ''' Parse arguments '''
    args = create_args()

    ''' Boot tachy-bs '''
    boot(args)

    ''' Save model to tachy-bs '''
    save_model(args)
    print("Model saved successfully")
    ''' Make instance for inference '''
    make_instance(args)
    print("Instance created successfully")
    ''' Connect to instance for inference '''
    connect_instance(args)
    print("Instance connected successfully")
    ''' Do inference '''
    inference(args)
    print("Inference done successfully")
    ''' Display result '''
    display(args)