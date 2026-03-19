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
import tachy_rt.core.functions as rt_core

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

    args.model_path = '../utils/object_detection_yolov9/{}/model_{}_inv-f.tachyrt'.format(args.model, args.input_shape)
    args.clss_dict  = read_json('../utils/object_detection_yolov9/{}/class.json'.format(args.model))
    args.images_input = glob.glob(f'{args.input_dir}/input*')
    args.images_input.sort()
    args.images_input = [cv2.resize(cv2.imread(f), (args.w,args.h)) for f in args.images_input]


    return args

def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return

    ''' Upload firmware to device '''
    spi_type = args.interface.split(":")[-1]
    ret = rt_core.boot(path=args.path_firmware, spi_type=spi_type)
    if ret:
        print("Success to boot. Check the status via uart or other api")
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)

def save_model(args):
    ''' Upload model '''
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
                "std": 255.0,
                "mean": 0.0
            }
        ],
        "output": {
            "reorder": True
        }
    }

    ''' Make engine '''
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
    args.predicts = []
    for i, img in enumerate(args.images_input):
        image = img.reshape(-1, args.h, args.w, 3)
        args.instance.process([[image]])
        args.ret = args.instance.get_result()

        args.post_config = read_json('../utils/object_detection_yolov9/{}/post_process_{}.json'.format(args.model, args.input_shape))
        sys.path.append('../utils/object_detection_yolov9/{}'.format(args.model))
        args.post = importlib.import_module('post_process').Decoder(args.post_config)
        args.anno = args.post.main(args.ret['buf'].view(np.float32), np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))

        for box in args.anno:
            _cls = args.clss_dict[str(box[1].astype(np.int32))]
            x0,y0,x1,y1 = box[2:6].astype(np.int32)
            img = cv2.rectangle(img.copy(), (x0,y0), (x1,y1), (255,0,0), 3)
            img = cv2.putText(img, _cls, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        args.predicts.append(img)

    rt_core.deinit_instance(args.interface, args.model_name)

def display(args):
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
    #boot(args)

    ''' Save model to tachy-bs '''
    save_model(args)

    ''' Make instance for inference '''
    make_instance(args)

    ''' Connect to instance for inference '''
    connect_instance(args)

    ''' Do inference '''
    inference(args)

    ''' Display result '''
    display(args)