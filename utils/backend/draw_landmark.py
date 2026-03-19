import os
import cv2
import numpy as np
from functions import draw_rectangle, draw_dot

def draw_landmark(img, anno):
    if len(anno) > 0:
        stc = anno[...,1:2] # 0 : open, 1 : close, 2 : ?
        box = anno[..., 2:6]
        dot_0 = anno[..., 6:8]
        dot_1 = anno[..., 8:10]

        img = draw_rectangle(img, box.astype(np.uint32))
        img = draw_dot(img, dot_0)
        img = draw_dot(img, dot_1)

    return img
