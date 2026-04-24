import os, sys
import re
sys.path.append('../utils/common')
import random
import commentjson
import string
import argparse
import numpy as np

import post_process

from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

REQUIRED = "3.2.2"

try:
    current = version("tachy-rt")
except PackageNotFoundError:
    raise RuntimeError("The tachy-rt package is not installed.")

if Version(current) < Version(REQUIRED):
    raise RuntimeError(
        f"Tachy Runtime >= {REQUIRED} is required (current: {current})."
    )

import tachy_rt.core.functions as rt_core
from threading import Thread

def read_json(name):
    with open(name) as f:
        _dict = commentjson.load(f)
        return _dict

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
    
        parser.add_argument('--model_path', type=str,
                            help='Model to inference',
                            required=True)

        parser.add_argument('--post_config_path', type=str,
                            help='post configs file path',
                            required=True)

        parser.add_argument('--input_shape', type=str,
                            help='Target input shape(HxWxD)',
                            default=None)

        parser.add_argument('--tx', type=int,
                            help='Target sensor tx number',
                            default=2)

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
                            default='true')

        parser.add_argument('--path_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='./tachy-shield')

        args = parser.parse_args()

        args.interface = "spi:host"

        return args
    
    args = get_parser()

    args.inverse_sync    = True if args.inverse_sync.lower() == 'true' else False
    args.inverse_data    = True if args.inverse_data.lower() == 'true' else False
    args.inverse_clock   = True if args.inverse_clock.lower() == 'true' else False
    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    args.h, args.w = map(int, re.search(r'_(\d+)x(\d+)x', args.model_path).groups())
    args.ratio = rt_core.get_sensor_ratio(args.h, args.w)
    args.model_name = "object_detection_yolov4"
    args.instance_name = id_generator()

    args.clss_dict  = read_json('./class.json')
    args.post_config = read_json('{}'.format(args.post_config_path))
    return args

def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return
    os.system("pinctrl set 4 op dl; sleep 3; pinctrl set 4 op dh")

    ''' Upload firmware to device '''

    data = {
        "spl" : {
            "path" : os.path.join(args.path_firmware, "spl.bin"),
            "addr" : "0x0"
        },
        "uboot" : {
            "path" : os.path.join(args.path_firmware, "u-boot.bin"),
            "addr" : "0x2000_0000"
        },
        "kernel" : {
            "path" : os.path.join(args.path_firmware, "image.ub"),
            "addr" : "0x4000_0000"
        },
        "fpga" : {
            "path" : os.path.join(args.path_firmware, "fpga_top.bin"),
            "addr" : "0x3000_0000"
        }}
    spi_type = args.interface.split(":")[-1]
    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        print("Success to boot.")
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)

def init_sensor(args):
    if not rt_core.enable_sensor(itf=args.interface, tx=args.tx, ratio_w=args.ratio, ratio_h=args.ratio, reset=True, inverse_data=args.inverse_data, inverse_sync=args.inverse_sync, inverse_clock=args.inverse_clock):
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
    args.post = post_process.Decoder(args.post_config)

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
