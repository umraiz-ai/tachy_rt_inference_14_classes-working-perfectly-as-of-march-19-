#!/usr/bin/env python3
import multiprocessing as mp
import os
import traceback


def _call_rt(queue, action, itf, name):
    try:
        import tachy_rt.core.functions as rt_core
        if action == "deinit":
            ret = rt_core.deinit_instance(itf, name)
        elif action == "delete":
            ret = rt_core.delete_model(itf, name)
        else:
            raise ValueError(f"Unknown action: {action}")
        queue.put(("ok", ret))
    except Exception:
        queue.put(("err", traceback.format_exc()))


def run_with_timeout(action, itf, name, timeout_sec=4):
    queue = mp.Queue()
    proc = mp.Process(target=_call_rt, args=(queue, action, itf, name), daemon=True)
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        return ("timeout", None)

    if queue.empty():
        return ("unknown", None)
    return queue.get()


def main():
    itf = os.environ.get("TACHY_INTERFACE", "spi:host")
    model_name = "object_detection_yolov9"
    instance_name = f"{model_name}_inst"

    print("Interface:", itf, flush=True)
    print("Running cleanup with per-call timeout", flush=True)

    for name in [instance_name, model_name]:
        status, data = run_with_timeout("deinit", itf, name)
        if status == "ok":
            print(f"deinit_instance ok: {name} -> {data}", flush=True)
        elif status == "timeout":
            print(f"deinit_instance timeout: {name}", flush=True)
        else:
            print(f"deinit_instance {status}: {name}", flush=True)
            if data:
                print(data, flush=True)

    status, data = run_with_timeout("delete", itf, model_name)
    if status == "ok":
        print(f"delete_model ok: {data}", flush=True)
    elif status == "timeout":
        print("delete_model timeout", flush=True)
    else:
        print(f"delete_model {status}", flush=True)
        if data:
            print(data, flush=True)

    print("Cleanup done.", flush=True)


if __name__ == "__main__":
    main()