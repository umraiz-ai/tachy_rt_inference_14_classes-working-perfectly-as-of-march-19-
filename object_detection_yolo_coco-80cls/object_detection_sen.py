#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.append('../utils/common')
import cv2
import time
import glob
import random
import string
import argparse
import numpy as np
import importlib
import tachy_rt.core.functions as rt_core

from threading import Thread
from functions import *

def start(args):
    cnt = 0
    while True:
        args.instance.process()
        cnt += 1

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

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

        parser.add_argument('--tx', type=int,
                            help='Target sensor tx number',
                            default=0)

        parser.add_argument('--inverse_data', type=str,
                            help='inverse data of sensor',
                            default='false')

        parser.add_argument('--inverse_sync', type=str,
                            help='inverse sync of sensor',
                            default='false')

        parser.add_argument('--inverse_clock', type=str,
                            help='inverse clock of sensor',
                            default='false')

        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='false')

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

    args.inverse_sync    = True if args.inverse_sync.lower() == 'true' else False
    args.inverse_data    = True if args.inverse_data.lower() == 'true' else False
    args.inverse_clock   = True if args.inverse_clock.lower() == 'true' else False
    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))
    args.ratio = rt_core.get_sensor_ratio(args.h, args.w)
    args.model_name = "object_detection_yolov9"
    args.instance_name = id_generator()

    args.model_path = '../utils/object_detection_yolov9/{}/model_{}_inv-f.tachyrt'.format(args.model, args.input_shape)
    args.clss_dict  = read_json('../utils/object_detection_yolov9/{}/class.json'.format(args.model))

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

def init_sensor(args):
    if not rt_core.enable_sensor(itf=args.interface, tx=args.tx, ratio=args.ratio, reset=True, inverse_data=args.inverse_data, inverse_sync=args.inverse_sync, inverse_clock=args.inverse_clock):
        print("Cannot enable sensor")
        print("please check your connection or sensor")
        print(rt_core.get_last_error_code())
        exit()

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
                "method": rt_core.INPUT_FMT_SENSOR,
                "std": 255.0,
                "mean": 0.0,
                "tx": args.tx
            }
        ],
        "output": {
            "reorder": True
        }
    }

    ''' Make engine '''
    ret = rt_core.make_instance(args.interface, args.model_name, args.instance_name, "frame_split", args.config)

    return ret

def connect_instance(args):
    ''' Connect instance '''
    ret, args.instance = rt_core.connect_instance(args.interface, args.instance_name)
    if not ret:
        print("connect instance fail")
        print("error :", rt_core.get_last_error_code())
        exit()

    return ret

def inference(args):
    ''' Create start thread '''
    start_thread = Thread(target=start, args=(args,))
    start_thread.start()

def get_result(args):
    ''' Load Input & Do inference'''
    args.post_config = read_json('../utils/object_detection_yolov9/{}/post_process_{}.json'.format(args.model, args.input_shape))
    sys.path.append('../utils/object_detection_yolov9/{}'.format(args.model))
    args.post = importlib.import_module('post_process').Decoder(args.post_config)

    while True:
        args.ret = args.instance.get_result()
        args.anno = args.post.main(args.ret['buf'].view(np.float32), np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))

        print(f"Detected Object : {len(args.anno)}")
        for obj in args.anno:
            prob,_cls,x0,y0,x1,y1 = obj
            _cls = args.clss_dict[str(int(_cls))]

            print(f"Class : {_cls}")
            print(f"Box   : {x0} {y0} {x1} {y1}")
        print("")

if __name__ == '__main__':
    ''' Parse arguments '''
    args = create_args()

    ''' Boot tachy-bs '''
    boot(args)

    ''' Init sensor interface'''
    init_sensor(args)

    ''' Save model to tachy-bs '''
    save_model(args)

    ''' Make instance for inference '''
    make_instance(args)

    ''' Connect to instance for inference '''
    connect_instance(args)

    ''' Do inference '''
    inference(args)

    ''' Get result'''
    get_result(args)
