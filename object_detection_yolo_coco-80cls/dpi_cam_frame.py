# filepath: /home/dpi/raspberrypi_20241209/inference/example/object_detection_yolo_coco-80cls/dpi_cam_frame.py

import cv2
import mmap
import numpy as np
from fcntl import ioctl
from v4l2 import v4l2_capability, v4l2_format, VIDIOC_QUERYCAP, VIDIOC_G_FMT, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_QUERYBUF, VIDIOC_QBUF
import v4l2
import time
from array import array

class V4L2Cam:
    def __init__(self, device, buffer_count=4):
        self.device = device
        self.fd = None
        self.buffer_count = buffer_count
        self.buffers = []
        self.width = 0
        self.height = 0

    def open(self):
        self.fd = open(self.device, 'r+b', buffering=0)

    def close(self):
        if self.fd is not None:
            self.fd.close()
            self.fd = None

    def init_cam(self, width=1920, height=1080):
        cap = v4l2.v4l2_capability()
        fmt = v4l2.v4l2_format()

        if (ioctl(self.fd, v4l2.VIDIOC_QUERYCAP, cap) < 0):
            print("querycap failed!")
            return -1
        if not (cap.capabilities & v4l2.V4L2_CAP_VIDEO_CAPTURE):
            print("video capture not supported!")
            return -1
        
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        if (ioctl(self.fd, v4l2.VIDIOC_G_FMT, fmt) < 0):
            print("get format failed!")
            return -1
        
        # request target resolution, then read back actual values
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YVYU
        if (ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt) < 0):
            print("set format failed!")
            return -1

        # save actual device-supported width/height for later use (avoid hardcoded reshape)
        self.width = int(fmt.fmt.pix.width)
        self.height = int(fmt.fmt.pix.height)

        print(f"Camera format: {self.width}x{self.height}, pixfmt={fmt.fmt.pix.pixelformat}")
        return 0

    def init_mmap(self):
        req = v4l2.v4l2_requestbuffers()
        req.count = self.buffer_count
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP

        if (ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req) < 0):
            print("request buffer failed!")
            return -1

        # get integer fileno for mmap
        fd_no = self.fd.fileno()

        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i

            if (ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf) < 0):
                print("query buffer failed!")
                return -1

            # create mmap using the integer file descriptor
            buf.buffer = mmap.mmap(fd_no, buf.length, access=mmap.ACCESS_READ, offset=buf.m.offset)
            self.buffers.append(buf)

        for buf in self.buffers:
            ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        return 0

    def start_capture(self):
        buf_type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        arg = array('i', [buf_type])
        # use fileno and pass a mutable buffer (array) so ioctl gets a pointer
        if (ioctl(self.fd.fileno(), v4l2.VIDIOC_STREAMON, arg) < 0):
            print("start capture failed!")
            return -1
        return 0

    def stop_capture(self):
        buf_type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        arg = array('i', [buf_type])
        ioctl(self.fd.fileno(), v4l2.VIDIOC_STREAMOFF, arg)

    def get_image(self):
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP

        # dequeue a buffer
        if (ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf) < 0):
            print("dequeue buffer failed!")
            return None

        # process the image (in this case, just convert YUYV to BGR)
        bgr_image = self.process_image(buf)

        # requeue the buffer
        ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        return bgr_image

    def process_image(self, buf):
        video_buffer = self.buffers[buf.index].buffer
        # read exactly the bytes used by the kernel for this frame
        data = video_buffer[:buf.bytesused]
        if len(data) == 0:
            print("DEBUG: empty buffer (bytesused=0) index:", buf.index)
            return None

        # use actual width/height (fallback to 1920x1080 if not set)
        h = getattr(self, "height", 1080)
        w = getattr(self, "width", 1920)
        expected = w * h * 2  # YVYU 16-bit per pixel (2 bytes)
        if len(data) < expected:
            print(f"DEBUG: data length {len(data)} smaller than expected {expected} (w={w},h={h})")

        try:
            yvyu_frame = np.frombuffer(data, np.uint8)
            # if kernel returned more/less bytes, allow reshape if divisible by 2*w; otherwise try best-effort reshape
            yvyu_frame = yvyu_frame[:(h*w*2)].reshape(h, w, 2)
            bgr_frame = cv2.cvtColor(yvyu_frame, cv2.COLOR_YUV2BGR_YVYU)
        except Exception as e:
            print("DEBUG: process_image failed:", e)
            return None

        return bgr_frame

if __name__ == "__main__":
    cam = V4L2Cam("/dev/video0")
    cam.open()
    if (cam.init_cam() < 0):
        exit(-1)
    if (cam.init_mmap() < 0):
        exit(-1)
    if (cam.start_capture() < 0):
        exit(-1)
    # cam.main_loop()
    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    while True:
        img = cam.get_image()
        if img is None:
            time.sleep(0.05)
            continue
        cv2.imshow('cam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop_capture()
    cam.close()
