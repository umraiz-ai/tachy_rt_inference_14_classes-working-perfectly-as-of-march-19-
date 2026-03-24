#!/usr/bin/env python
# coding: utf-8
"""
Object detection (sensor input) for Tachy-RT 3.2.2+.

Uses: boot(spi_type, DEV_TACHY_SHIELD, data dict), enable_sensor(..., ratio_w, ratio_h),
make_instance with frame_spliter / frame_splitter fallback, connect_instance.
"""

import os
import sys

sys.path.append('../utils/common')
import argparse
import importlib
import random
import string
import numpy as np
import tachy_rt.core.functions as rt_core
from threading import Thread

from functions import read_json


def start(args):
    while True:
        args.instance.process()


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def _build_boot_data(path_firmware: str):
    spl = os.path.join(path_firmware, "spl.bin")
    uboot = os.path.join(path_firmware, "u-boot.bin")
    if not os.path.exists(uboot):
        alt = os.path.join(path_firmware, "uboot.bin")
        if os.path.exists(alt):
            uboot = alt

    kernel = os.path.join(path_firmware, "image.ub")

    fpga = os.path.join(path_firmware, "fpga_top.bin")
    if not os.path.exists(fpga):
        alt = os.path.join(path_firmware, "fpga.bin")
        if os.path.exists(alt):
            fpga = alt

    required = [spl, uboot, kernel, fpga]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing firmware files for tachy_rt 3.2.2 boot():")
        for m in missing:
            print(" -", m)
        return None

    return {
        "spl": {"path": spl, "addr": "0x0"},
        "uboot": {"path": uboot, "addr": "0x2000_0000"},
        "kernel": {"path": kernel, "addr": "0x4000_0000"},
        "fpga": {"path": fpga, "addr": "0x3000_0000"},
    }


def create_args():
    parser = argparse.ArgumentParser(description='Deep learning example script (Tachy-RT 3.2.2 sensor)')

    parser.add_argument('--model', type=str, help='Subfolder under utils/object_detection_yolov9/', required=True)
    parser.add_argument('--input_shape', type=str, help='Target input shape HxWxD', required=True)
    parser.add_argument('--tx', type=int, help='Sensor TX channel', default=0)
    parser.add_argument('--inverse_data', type=str, default='false')
    parser.add_argument('--inverse_sync', type=str, default='false')
    parser.add_argument('--inverse_clock', type=str, default='false')
    parser.add_argument('--upload_firmware', type=str, default='false')
    parser.add_argument(
        '--path_firmware',
        type=str,
        default='../firmware/tachy-shield',
        help='Firmware directory for boot()',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Optional full path to .tachyrt (default: ../utils/object_detection_yolov9/<model>/model_<input_shape>_inv-f.tachyrt)',
    )
    parser.add_argument(
        '--class_path',
        type=str,
        default=None,
        help='Optional full path to class.json',
    )
    parser.add_argument(
        '--post_config_path',
        type=str,
        default=None,
        help='Optional full path to post_process_<shape>.json',
    )

    args = parser.parse_args()

    if "TACHY_INTERFACE" not in os.environ:
        print('Environment "TACHY_INTERFACE" is not set')
        sys.exit(1)

    args.interface = os.environ["TACHY_INTERFACE"]
    args.inverse_sync = args.inverse_sync.lower() == 'true'
    args.inverse_data = args.inverse_data.lower() == 'true'
    args.inverse_clock = args.inverse_clock.lower() == 'true'
    args.upload_firmware = args.upload_firmware.lower() == 'true'
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))

    # 3.2.2: separate ratio_w / ratio_h (same value from helper, matching main__example_file_for_3.2.2.py)
    args.ratio = rt_core.get_sensor_ratio(args.h, args.w)
    args.ratio_w = args.ratio
    args.ratio_h = args.ratio

    args.model_name = "object_detection_yolov9"
    args.instance_name = id_generator()

    if args.model_path:
        args.model_path_resolved = os.path.abspath(args.model_path)
    else:
        args.model_path_resolved = os.path.abspath(
            '../utils/object_detection_yolov9/{}/model_{}_inv-f.tachyrt'.format(
                args.model, args.input_shape
            )
        )

    if args.class_path:
        args.class_path_resolved = os.path.abspath(args.class_path)
    else:
        args.class_path_resolved = os.path.abspath(
            '../utils/object_detection_yolov9/{}/class.json'.format(args.model)
        )
    args.clss_dict = read_json(args.class_path_resolved)

    if args.post_config_path:
        args.post_config_path_resolved = os.path.abspath(args.post_config_path)
    else:
        args.post_config_path_resolved = os.path.abspath(
            '../utils/object_detection_yolov9/{}/post_process_{}.json'.format(
                args.model, args.input_shape
            )
        )
    return args


def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return True

    spi_type = args.interface.split(":")[-1]
    data = _build_boot_data(os.path.abspath(args.path_firmware))
    if data is None:
        return False

    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        print("Success to boot. Check the status via uart or other api")
        return True
    print("Failed to boot")
    print("Error code :", rt_core.get_last_error_code())
    return False


def init_sensor(args):
    ok = rt_core.enable_sensor(
        itf=args.interface,
        tx=args.tx,
        ratio_w=args.ratio_w,
        ratio_h=args.ratio_h,
        reset=True,
        inverse_data=args.inverse_data,
        inverse_sync=args.inverse_sync,
        inverse_clock=args.inverse_clock,
    )
    if not ok:
        print("Cannot enable sensor")
        print("please check your connection or sensor")
        print(rt_core.get_last_error_code())
        sys.exit(1)


def save_model(args):
    ret = rt_core.save_model(
        args.interface,
        args.model_name,
        rt_core.MODEL_STORAGE_MEMORY,
        args.model_path_resolved,
        overwrite=True,
    )
    return ret


def make_instance(args):
    try:
        rt_core.deinit_instance(args.interface, args.instance_name)
    except Exception:
        pass
    try:
        rt_core.deinit_instance(args.interface, args.model_name)
    except Exception:
        pass

    args.config = {
        "global": {
            "name": args.model_name,
            "data_type": rt_core.DTYPE_FLOAT16,
            "buf_num": 5,
            "max_batch": 1,
            "npu_mask": -1,
        },
        "input": [
            {
                "method": rt_core.INPUT_FMT_SENSOR,
                "std": 255.0,
                "mean": 0.0,
                "tx": args.tx,
            }
        ],
        "output": {"reorder": True},
    }

    for algo in ("frame_spliter", "frame_splitter"):
        ret = rt_core.make_instance(
            args.interface,
            args.model_name,
            args.instance_name,
            algo,
            args.config,
        )
        if ret:
            print(f"make_instance success with algorithm: {algo}")
            return True

    print("make_instance fail")
    print("error :", rt_core.get_last_error_code())
    return False


def connect_instance(args):
    ret, args.instance = rt_core.connect_instance(args.interface, args.instance_name)
    if not ret:
        print("connect instance fail")
        print("error :", rt_core.get_last_error_code())
        sys.exit(1)
    return ret


def inference(args):
    start_thread = Thread(target=start, args=(args,), daemon=True)
    start_thread.start()


def get_result(args):
    args.post_config = read_json(args.post_config_path_resolved)
    sys.path.append('../utils/object_detection_yolov9/{}'.format(args.model))
    args.post = importlib.import_module('post_process').Decoder(args.post_config)

    while True:
        args.ret = args.instance.get_result()
        args.anno = args.post.main(
            args.ret['buf'].view(np.float32),
            np.array([[0, 0, args.w - 1, args.h - 1]], dtype=np.float32),
        )

        print(f"Detected Object : {len(args.anno)}")
        for obj in args.anno:
            prob, _cls, x0, y0, x1, y1 = obj
            _cls = args.clss_dict[str(int(_cls))]
            print(f"Class : {_cls}")
            print(f"Box   : {x0} {y0} {x1} {y1}")
        print("")


if __name__ == '__main__':
    args = create_args()

    if not boot(args):
        sys.exit(-1)

    init_sensor(args)

    if not save_model(args):
        print("save_model fail")
        print("Error :", rt_core.get_last_error_code())
        sys.exit(-1)

    if not make_instance(args):
        sys.exit(-1)

    connect_instance(args)
    inference(args)
    get_result(args)
