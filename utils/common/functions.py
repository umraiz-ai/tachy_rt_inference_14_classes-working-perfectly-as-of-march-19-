import os
import cv2
import commentjson
import numpy as np
# import requests

def _align_dim(pos):
    if type(pos) == list:
        pos = np.array(pos)

    if type(pos) == np.ndarray:
        if pos.ndim == 1: pos = pos[None, :]

    return pos

def draw_rectangle(img, pos, color=(255,0,0), thickness=3):
    pos = _align_dim(pos)
    for box in pos:
        img = cv2.rectangle(img.astype(np.uint8), (box[0], box[1]), (box[2], box[3]), color, thickness)

    return img

def draw_dot(img, pos, color=(0, 255, 0), radian=2, thickness=2):
    pos = _align_dim(pos)

    for p in pos:
        p = list(map(round, p))
        p = list(map(int, p))
        img = cv2.circle(img, (p[0], p[1]), radian, color, thickness)
    return img

def put_txt(img, txt, x, y, scale=1.0, color=(0,0,255), t=2):
    img = cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)
    return img

def display_img(img, wait_time):    
    print(img.shape, img.dtype)
    cv2.imshow('', img)
    cv2.waitKey(wait_time)

def check_file(name, show=True):
    # Check if file exist
    if not os.path.isfile(name):
        raise OSError("File doesn't exist: {}".format(name))
    if show:
        print("  {} file is exist".format(name))

def read_json(name):
    with open(name) as f:
        _dict = commentjson.load(f)
        return _dict
