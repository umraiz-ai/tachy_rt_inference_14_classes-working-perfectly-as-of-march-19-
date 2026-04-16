# Tachy RT 3.2.2 API Documentation

This document provides a comprehensive reference for the `tachy-rt` version 3.2.2. It is designed to be used by developers and AI coding assistants (like Cursor AI) for implementing high-performance NPU inference.

---

## 1. Core Lifecycle API
Modules: `import tachy_rt.core.functions as rt_core`

### `boot(spi_type: str, device_type: int, data: dict) -> bool`
Initializes and uploads firmware to the Tachy device.
- **`spi_type`**: `"ftdi"` (for FTDI adapters) or `"host"` (for native SPI, e.g., Raspberry Pi).
- **`device_type`**: `rt_core.DEV_TACHY_SHIELD`.
- **`data`**: A dictionary requiring specific keys and hex addresses:
  ```python
  data = {
      "spl":    {"path": "path/to/spl.bin",    "addr": "0x0"},
      "uboot":  {"path": "path/to/u-boot.bin", "addr": "0x2000_0000"},
      "kernel": {"path": "path/to/image.ub",   "addr": "0x4000_0000"},
      "fpga":   {"path": "path/to/fpga_top.bin","addr": "0x3000_0000"}
  }
  ```

### `save_model(itf: str, name: str, storage: int, model_path: str, overwrite: bool) -> bool`
Loads and stores a `.tachyrt` model on the device.
- **`storage`**: `rt_core.MODEL_STORAGE_MEMORY` or `rt_core.MODEL_STORAGE_NOR`.
- **`itf`**: Interface string (e.g., `"spi:host"`).

---

## 2. Sensor API
Used for live camera feeds via the Tachy hardware.

### `enable_sensor(itf: str, tx: int, ratio_w: int=1, ratio_h: int=1, crop: list=[0,0,1919,1079], reset: bool=False, ...)`
- **`tx`**: The sensor channel (0-3).
- **`ratio_w` / `ratio_h`**: Downscaling factors. If the camera is 1080p and you want 416 width, set ratio accordingly.
- **`reset`**: Set to `True` if re-initializing the sensor.

---

## 3. Inference Instance API
Standard workflow: `make_instance` -> `connect_instance` -> `instance.process`.

### `make_instance(itf: str, model_name: str, instance_name: str, algorithm: str, config: dict) -> bool`
- **`algorithm`**: Always `"frame_spliter"`.
- **`config` Structure**:
  ```python
  config = {
      "global": {
          "name": model_name,
          "data_type": rt_core.DTYPE_FLOAT16,
          "buf_num": 5, # Recommended buffer count
          "max_batch": 1,
          "npu_mask": -1 # Use all NPUs
      },
      "input": [{
          "method": rt_core.INPUT_FMT_SENSOR, # or INPUT_FMT_BINARY
          "std": 255.0,
          "mean": 0.0
      }],
      "output": {"reorder": True}
  }
  ```

### `connect_instance(itf: str, instance_name: str) -> (bool, instance_object)`
Returns a tuple. If successful, the second element is the handle used for inference.

### `instance.process(data=None)`
- **`data`**: List of images `[[numpy_array]]` for `INPUT_FMT_BINARY`. Set to `None` for `INPUT_FMT_SENSOR`.
- **Non-blocking**: This sends a request to the NPU and returns immediately.

### `instance.get_result() -> dict`
- **Blocking**: Waits for the next available result from the NPU.
- **Returns**: `{"buf": np.ndarray, "index": int, ...}`.

---

## 4. Key Differences (3.2.0 vs 3.2.2)

| Feature | Version 3.2.0 | Version 3.2.2 (New) |
| :--- | :--- | :--- |
| **Boot API** | `boot(path="...", ...)` | `boot(spi_type, dev, data_dict)` (Dict based paths) |
| **Sensor Ratios** | `ratio` (single int) | `ratio_w`, `ratio_h` (separate ints) |
| **Instance Return** | Returns Object or None | Returns `bool` (success/failure) |
| **Connection** | Integrated in make | `connect_instance` required after `make` |
| **Threading** | Mostly synchronous examples | Async pattern (Threaded `process`) enforced |

---

## 5. Architectural Pattern: Async Inference
To achieve maximum FPS in **Sensor Mode**, you **must** use a producer-consumer pattern with threads:

```python
from threading import Thread

# Thread 1: Producer (Keep NPU busy)
def inference_thread(inst):
    while True:
        inst.process() # Send next frame request asynchronously

# Thread 2: Consumer (Main Loop)
Thread(target=inference_thread, args=(instance,), daemon=True).start()

while True:
    result = instance.get_result() # Wait for results and free buffer
    # Post-process result['buf']
```

## 6. Important Constants
- `rt_core.DTYPE_FLOAT16` = 0
- `rt_core.INPUT_FMT_BINARY` = 0
- `rt_core.INPUT_FMT_SENSOR` = 1
- `rt_core.MODEL_STORAGE_MEMORY` = 0
- `rt_core.DEV_TACHY_SHIELD` = 0
