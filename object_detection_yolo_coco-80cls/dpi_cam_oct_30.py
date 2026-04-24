import os
import struct 
import time

import cv2
from fcntl import ioctl
import mmap
import numpy as np
import v4l2 as v4l2


class V4L2Cam:
    def __init__(self, device_path):
        self.fd = os.open(device_path, os.O_RDWR, 0)
        self.buffers = []
        self.buffer_count = 1
        self.cnt = 0

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
        
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height

        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YVYU
        if (ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt) < 0):
            print("set format failed!")
            return -1

        return 0

    def init_mmap(self):
        req = v4l2.v4l2_requestbuffers()
        req.count = self.buffer_count
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP

        if (ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req) < 0):
            print("request buffer failed!")
            return -1

        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i

            if (ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf) < 0):
                print("query buffer failed!")
                return -1

            # self.buffers.append(mmap.mmap(self.fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
            buf.buffer = mmap.mmap(self.fd, buf.length, mmap.PROT_READ, mmap.MAP_SHARED, offset=buf.m.offset)
            self.buffers.append(buf)

        for buf in self.buffers:
            ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        return 0

    def start_capture(self):
        try:
            ioctl(self.fd, v4l2.VIDIOC_STREAMON, struct.pack('I', v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
            print("VIDIOC_STREAMON executed successfully")
        except OSError as e:
            print(f"Error: VIDIOC_STREAMON failed with error: {e}")
        return 0

    def dqbuf(self, index):
        if (ioctl(self.fd, v4l2.VIDIOC_DQBUF, self.buffers[index]) < 0):
            print("dequeue buffer failed!")
            return -1

    def qbuf(self, index):
        if (ioctl(self.fd, v4l2.VIDIOC_QBUF, self.buffers[index]) < 0):
            print("queue buffer failed!")
            return -1

    def process_image(self, buf):
        video_buffer = self.buffers[buf.index].buffer
        data = video_buffer.read()[:buf.bytesused]
        yvyu_frame = np.frombuffer(data, np.uint8).reshape(1080, 1920, 2)
        bgr_frame = cv2.cvtColor(yvyu_frame, cv2.COLOR_YUV2BGR_YVYU)
        video_buffer.seek(0)
        return bgr_frame

    def main_loop(self):
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for i in range(100):
            start = time.time()
            buf = self.buffers[i % self.buffer_count]
            if (ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf) < 0):
                print("dequeue buffer failed!")
                return -1
            self.process_image(buf)
            ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)
            print("Frame: ", i, ", buf index: ", i % self.buffer_count, "Time: ", time.time() - start)

    def get_image(self):
        buf = self.buffers[self.cnt % self.buffer_count]
        if (ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf) < 0):
            print("dequeue buffer failed!")
            return -1
        img = self.process_image(buf)
        ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)
        self.cnt += 1
        return img

if __name__ == "__main__":
    cam = V4L2Cam("/dev/video0")
    if (cam.init_cam() < 0):
        exit(-1)
    if (cam.init_mmap() < 0):
        exit(-1)
    if (cam.start_capture() < 0):
        exit(-1)
    # cam.main_loop()
    while 1:
        img = cam.get_image()
        cv2.imshow('', img)
        cv2.waitKey(0)
