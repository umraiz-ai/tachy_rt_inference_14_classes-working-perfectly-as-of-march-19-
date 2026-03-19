import os
import cv2
import numpy as np
from functions import draw_rectangle, put_txt

def draw_box(image, ants, labels=None, txt=True, thickness=3):
    if labels is None:
        for ant in ants:
            image = draw_rectangle(image, ant[2:], color=(0,0,255), thickness=thickness)
    else:
        for ant in ants:
            prob = round(float(ant[0])*100, 2)
            cls = str(int(ant[1]))
            box = ant[2:]
            box = list(map(round, box))
            box = list(map(int, box))
            image = draw_rectangle(image, box, color=tuple(labels[cls]['COL']), thickness=thickness)
            if txt: image = put_txt(image, '{}: {}'.format(labels[cls]['CLS'], prob), box[0]+2, box[3]+30, color=tuple(labels[cls]['COL']), scale=0.75)

    return image
