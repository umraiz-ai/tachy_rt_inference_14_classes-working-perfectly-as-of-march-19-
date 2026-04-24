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

from functions import *


def create_args():
    def get_parser():
        """
        Get the parser.
        :return: parser
        """
        parser = argparse.ArgumentParser(description='Deep learning example script')

        # parser.add_argument('--model', type=str,
        #                     help='Model to inference',
        #                     required=True)

        parser.add_argument('--input_shape', type=str,
                            help='Target input shape(HxWxD)',
                            default=None)

        parser.add_argument('--input_dir', type=str,
                            help='Directory for inputs',
                            required=True)

        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='false')

        parser.add_argument('--path_firmware', type=str,
                            help='Firmware directory path (spi interface only)',
                            default='../firmware')

        args = parser.parse_args()

        ENVS = os.environ
        if "TACHY_INTERFACE" not in ENVS:
            print('Environment "TACHY_INTERFACE" is not set')
            exit(-1)

        args.interface = ENVS["TACHY_INTERFACE"]
        return args

    args = get_parser()

    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    args.h, args.w = list(map(int, args.input_shape.split('x')[:2]))
    args.model_name = "object_detection_yolov9"

    args.model_path = '/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/1_Mar_18_14_model_416x416x3_inv-f.tachyrt'
    args.clss_dict = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/class.json')
    args.images_input = glob.glob(f'{args.input_dir}/input*')
    args.images_input.sort()
    args.images_input = [cv2.resize(cv2.imread(f), (args.w, args.h)) for f in args.images_input]

    return args


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
        "spl":    {"path": spl,    "addr": "0x0"},
        "uboot":  {"path": uboot,  "addr": "0x2000_0000"},
        "kernel": {"path": kernel, "addr": "0x4000_0000"},
        "fpga":   {"path": fpga,   "addr": "0x3000_0000"},
    }


def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return True

    # Upload firmware to device (tachy_rt 3.2.2 API)
    spi_type = args.interface.split(":")[-1]
    data = _build_boot_data(args.path_firmware)
    if data is None:
        return False

    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        print("Success to boot. Check the status via uart or other api")
        return True

    print("Failed to boot")
    print("Error code :", rt_core.get_last_error_code())
    return False


def save_model(args):
    # Upload model
    ret = rt_core.save_model(
        args.interface,
        args.model_name,
        rt_core.MODEL_STORAGE_MEMORY,
        args.model_path,
        overwrite=True
    )
    return ret


def make_instance(args):
    # Clean stale instance (ignore errors)
    try:
        rt_core.deinit_instance(args.interface, f"{args.model_name}_inst")
    except Exception:
        pass
    try:
        rt_core.deinit_instance(args.interface, args.model_name)
    except Exception:
        pass

    args.instance_name = f"{args.model_name}_inst"

    args.config = {
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
        "output": {"reorder": True}
    }

    # Try both tokens (package variants)
    for algo in ("frame_spliter", "frame_splitter"):
        ret = rt_core.make_instance(
            args.interface,
            args.model_name,
            args.instance_name,
            algo,
            args.config
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
        return False
    return True


def inference(args):
    # Prepare post-process once
    post_config = read_json('/home/dpi/Desktop/inference__nov_migration/example/utils/object_detection_yolov9/req_files_ppr/post_process_416x416x3.json')
    sys.path.append('../utils/object_detection_yolov9/req_files_ppr')
    post = importlib.import_module('post_process').Decoder(post_config)

    # Load input & do inference
    args.predicts = []
    for img in args.images_input:
        image = img.reshape(-1, args.h, args.w, 3)
        args.instance.process([[image]])
        ret = args.instance.get_result()
        # DEBUG: run once per image (e.g. first image only) to see score range
        buf = ret['buf'].view(np.float32).ravel()
        n_grid = post.n_grid
        n_box, n_cls = 4, post.n_cls   # match config
        box_elems = n_grid * n_box
        cls_flat = buf[box_elems:]
        cls_pred = np.reshape(cls_flat, (-1, n_cls))
        from operations import sigmoid
        probs = sigmoid(cls_pred)
        max_per_cell = np.max(probs, axis=1)
        print("max class prob (any cell):", float(np.max(max_per_cell)))
        print("cells above 0.01 / 0.1 / 0.25:", np.sum(max_per_cell > 0.01), np.sum(max_per_cell > 0.1), np.sum(max_per_cell > 0.25))
        print("buf size, n_grid, floats/grid:", buf.size, n_grid, buf.size / n_grid)
        anno = post.main(ret['buf'].view(np.float32), np.array([[0, 0, args.w - 1, args.h - 1]], dtype=np.float32))

        draw = img.copy()
        for box in anno:
            _cls = args.clss_dict[str(box[1].astype(np.int32))]
            x0, y0, x1, y1 = box[2:6].astype(np.int32)
            draw = cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 0, 0), 3)
            draw = cv2.putText(draw, _cls, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        args.predicts.append(draw)

    rt_core.deinit_instance(args.interface, args.instance_name)


def display(args):
    inputs = np.concatenate(args.images_input, axis=0)
    predicts = np.concatenate(args.predicts, axis=0)
    total_img = np.concatenate([inputs, predicts], axis=1)
    cv2.imwrite('result.png', total_img)


if __name__ == '__main__':
    # Parse arguments
    args = create_args()

    # Boot tachy-bs (optional)
    if not boot(args):
        exit(-1)

    # Save model to tachy-bs
    if not save_model(args):
        print("save_model fail")
        print("error :", rt_core.get_last_error_code())
        exit(-1)

    # Make instance for inference
    if not make_instance(args):
        print("make_instance fail")
        print("error :", rt_core.get_last_error_code())
        exit(-1)

    # Connect to instance for inference
    if not connect_instance(args):
        exit(-1)

    # Do inference
    inference(args)

    # Display result
    display(args)