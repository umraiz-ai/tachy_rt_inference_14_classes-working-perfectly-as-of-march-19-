#!/usr/bin/env python3
# coding: utf-8
"""
Minimal example: invoke the NPU via Tachy-RT (tachy_rt.core.functions).

The NPU runs when you call instance.process(...) after save_model + make_instance
+ connect_instance. There is no separate low-level NPU API in user code.

Prerequisites:
  export TACHY_INTERFACE="spi:host"    # or spi:ftdi, ethernet:IP, local, etc.

Usage:
  python3 npu_invoke_example.py \\
    --model /path/to/model.tachyrt \\
    --input_shape 416x416x3

  python3 npu_invoke_example.py \\
    --model /path/to/model.tachyrt \\
    --input_shape 416x416x3 \\
    --image /path/to/image.jpg

Stuck after "init tachyrt_model v2.0"?
  That log is from parsing the .tachyrt file. The next step is save_model(), which
  talks to the board over TACHY_INTERFACE. If there is no response (SPI wiring,
  wrong interface string, board off, or cold SPI without boot), it can block forever.
  Try: correct TACHY_INTERFACE, --upload_firmware true (SPI), check /dev/spidev*, ping (Ethernet).

The turbojpeg warning is harmless.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import tachy_rt.core.functions as rt_core


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


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
        print("Missing firmware files for boot():")
        for m in missing:
            print(" -", m)
        return None

    return {
        "spl": {"path": spl, "addr": "0x0"},
        "uboot": {"path": uboot, "addr": "0x2000_0000"},
        "kernel": {"path": kernel, "addr": "0x4000_0000"},
        "fpga": {"path": fpga, "addr": "0x3000_0000"},
    }


def boot_spi_if_needed(
    interface: str,
    upload_firmware: bool,
    path_firmware: str,
    verbose: bool = True,
) -> bool:
    if "spi" not in interface or not upload_firmware:
        _log("[npu_invoke] skip boot (not SPI or --upload_firmware false)", verbose)
        return True

    _log("[npu_invoke] calling boot() …", verbose)
    spi_type = interface.split(":")[-1]
    data = _build_boot_data(path_firmware)
    if data is None:
        return False

    ok = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ok:
        _log("[npu_invoke] boot(): success", verbose)
        return True
    print("boot(): failed, error:", rt_core.get_last_error_code(), flush=True)
    return False


def make_runtime_config(model_name: str, npu_mask: int = -1) -> dict:
    # npu_mask: -1 = use all available tensor cores (per Tachy-RT docs)
    return {
        "global": {
            "name": model_name,
            "data_type": rt_core.DTYPE_FLOAT16,
            "buf_num": 5,
            "max_batch": 1,
            "npu_mask": npu_mask,
        },
        "input": [
            {
                "method": rt_core.INPUT_FMT_BINARY,
                "std": 255.0,
                "mean": 0.0,
            }
        ],
        "output": {"reorder": True},
    }


def run_npu_inference(
    interface: str,
    model_path: str,
    model_name: str,
    height: int,
    width: int,
    image: np.ndarray | None = None,
    upload_firmware: bool = False,
    path_firmware: str = "./firmware/tachy-shield",
    npu_mask: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Full pipeline: optional boot -> save_model -> make_instance -> connect ->
    process (NPU) -> get_result.

    Returns raw output as float32 view of the result buffer (same as other scripts).
    """
    _log(f"[npu_invoke] TACHY_INTERFACE={interface!r}", verbose)
    if not boot_spi_if_needed(
        interface, upload_firmware, path_firmware, verbose=verbose
    ):
        raise RuntimeError("boot failed")

    try:
        sz = os.path.getsize(model_path)
    except OSError:
        sz = -1
    _log(
        "[npu_invoke] calling save_model() … (parses .tachyrt then uploads to device; "
        "can block here if the link to the NPU is down)",
        verbose,
    )
    _log(f"[npu_invoke] model file: {model_path} size_bytes={sz}", verbose)

    if not rt_core.save_model(
        interface,
        model_name,
        rt_core.MODEL_STORAGE_MEMORY,
        model_path,
        overwrite=True,
    ):
        raise RuntimeError(
            f"save_model failed: {rt_core.get_last_error_code()}"
        )
    _log("[npu_invoke] save_model() returned OK", verbose)

    instance_name = f"{model_name}_inst"
    try:
        rt_core.deinit_instance(interface, instance_name)
    except Exception:
        pass
    try:
        rt_core.deinit_instance(interface, model_name)
    except Exception:
        pass

    config = make_runtime_config(model_name, npu_mask=npu_mask)
    ok = False
    _log("[npu_invoke] calling make_instance() …", verbose)
    for algo in ("frame_spliter", "frame_splitter"):
        ok = rt_core.make_instance(
            interface, model_name, instance_name, algo, config
        )
        if ok:
            _log(f"[npu_invoke] make_instance OK (algorithm={algo})", verbose)
            break
    if not ok:
        raise RuntimeError(
            f"make_instance failed: {rt_core.get_last_error_code()}"
        )

    _log("[npu_invoke] calling connect_instance() …", verbose)
    ret, instance = rt_core.connect_instance(interface, instance_name)
    if not ret or instance is None:
        raise RuntimeError(
            f"connect_instance failed: {rt_core.get_last_error_code()}"
        )
    _log("[npu_invoke] connect_instance OK", verbose)

    if image is None:
        # Synthetic input; replace with real preprocessed image for real tests
        image = np.zeros((1, height, width, 3), dtype=np.uint8)
    else:
        if image.shape != (1, height, width, 3):
            raise ValueError(
                f"image must be (1, {height}, {width}, 3), got {image.shape}"
            )

    # This call schedules execution on the NPU
    _log("[npu_invoke] calling instance.process() (NPU run) …", verbose)
    instance.process([[image]])
    _log("[npu_invoke] calling get_result() …", verbose)
    out = instance.get_result()
    buf = out["buf"].view(np.float32)

    _log("[npu_invoke] calling deinit_instance() …", verbose)
    rt_core.deinit_instance(interface, instance_name)
    return np.asarray(buf)


def parse_args():
    p = argparse.ArgumentParser(description="Invoke NPU once via Tachy-RT")
    p.add_argument("--model", required=True, help="Path to .tachyrt model")
    p.add_argument(
        "--model_name",
        default="object_detection_yolov9",
        help="Logical model name (must match what you use in save_model)",
    )
    p.add_argument(
        "--input_shape",
        required=True,
        help="HxWxC e.g. 416x416x3",
    )
    p.add_argument(
        "--image",
        default=None,
        help="Optional BGR image path (OpenCV). Resized to model input.",
    )
    p.add_argument(
        "--upload_firmware",
        choices=("true", "false"),
        default="false",
        help="SPI only: upload firmware before inference",
    )
    p.add_argument(
        "--path_firmware",
        default="./firmware/tachy-shield",
        help="Firmware directory for boot()",
    )
    p.add_argument(
        "--npu_mask",
        type=int,
        default=-1,
        help="NPU core mask (-1 = all cores, per Tachy-RT)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable step-by-step progress messages",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if "TACHY_INTERFACE" not in os.environ:
        print('Set environment variable TACHY_INTERFACE, e.g. export TACHY_INTERFACE="spi:host"')
        return 1

    interface = os.environ["TACHY_INTERFACE"]
    verbose = not args.quiet
    _log(f"[npu_invoke] starting (quiet={args.quiet})", verbose)
    parts = args.input_shape.lower().split("x")
    if len(parts) < 2:
        print("input_shape must be like 416x416x3")
        return 1
    h, w = int(parts[0]), int(parts[1])

    image = None
    if args.image:
        try:
            import cv2
        except ImportError:
            print("Install opencv-python or run without --image")
            return 1
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}")
            return 1
        img = cv2.resize(img, (w, h))
        image = img.reshape(1, h, w, 3)

    upload_fw = args.upload_firmware.lower() == "true"

    try:
        out = run_npu_inference(
            interface=interface,
            model_path=args.model,
            model_name=args.model_name,
            height=h,
            width=w,
            image=image,
            upload_firmware=upload_fw,
            path_firmware=args.path_firmware,
            npu_mask=args.npu_mask,
            verbose=verbose,
        )
    except RuntimeError as e:
        print("Error:", e)
        return 1

    print("NPU inference finished.")
    print("Output buffer shape:", out.shape, "dtype:", out.dtype)
    print(
        "Output stats: min={:.6f} max={:.6f} mean={:.6f}".format(
            float(out.min()), float(out.max()), float(out.mean())
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
