#coding:utf-8

import numpy as np
# from PIL import Image
# import cv2

# EPSILON_16 = 10e-5
# EPSILON_32 = 10e-9

EPSILON_16 = 9.77e-04
EPSILON_32 = 1.19e-07
EPSILON_64 = 2.22e-16

CLS_MAGIC_NUM = -100.0
LOC_MAGIC_NUM = -10000.0

NPY_DTYPE_DICT={
    'float16'   :np.float16,
    'float32'   :np.float32,
    'float64'   :np.float64,
    'uint8'     :np.uint8,
    'int8'      :np.int8,
    'int32'     :np.int32,
}

API = {
    0:'CV2',
    1:'PIL',
}
FLIP = {
    0:'NO_FLIP',
    1:'FLIP',
}
INTERPOLATION = {
    # 0:{
        # 0:cv2.INTER_NEAREST,
        # 1:cv2.INTER_LINEAR,
        # 2:cv2.INTER_CUBIC,
    # },
    # 1:{
        # 0:Image.NEAREST,
        # 1:Image.BILINEAR,
        # # 2:Image.CUBIC,
    # }
}

SCALE = {
    0:'KEEP_RATIO',
    1:'FIXED_RATIO',
}

DATA_LEN_DICT={
    'class' :1,
    'box'   :4,
    'line'  :4,
    'dot'   :2,
}

SHIFT_DICT = {
    np.uint8:0,
    np.int8:0,
    np.float16:1,
    np.uint16:1,
    np.int16:1,
    np.float32:2,
    np.uint32:2,
    np.int32:2,
    np.uint64:3,
    np.int64:3,
}

SHIFT_DICT_STR = {
    'uint8':0,
    'int8':0,
    'bool':0,
    'float16':1,
    'uint16':1,
    'int16':1,
    'float32':2,
    'uint32':2,
    'int32':2,
    'uint64':3,
    'int64':3,
    '|S128':7,
    '|S256':8,
}
