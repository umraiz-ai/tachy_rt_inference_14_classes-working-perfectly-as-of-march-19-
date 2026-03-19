#coding:utf-8

"""
Title: Object Detection YOLOv9, Postprocess and Methods(Encoder, Decoder)
Author: [Myungkyum Kim](dean@deeper-i.com)
Model: objcet_detection_yolov9
Description: Utilities for Yolov9 Decode
History:
    2024/08/21: init
"""


import os, sys
sys.path.append('libs')

import numpy as np
import cv2

from operations import sigmoid, py_nms, convert_to_xywh
# from cython_operations import nms_v2

#############
#### Methods
#############
class Encoder:
    def __init__(self, output_shapes, grid_sizes, n_batch=10):
        self.output_shapes = output_shapes
        self.grid_sizes = grid_sizes
        self.n_batch = n_batch

    def _compute_grid_base_matrix(self, output_shape):
        h, w, _ = output_shape
        if h == w:
            matrix = np.stack(np.meshgrid(range(h), range(w)), axis=-1) 
            matrix = matrix[:, :, None, :].astype('float32')
        else:
            x = np.tile(np.array(range(w)), (h,1)) + 0.5
            y = np.tile(np.array(range(h)), (w,1)).T + 0.5  # (H,W)
            matrix = np.concatenate([x[..., None], y[..., None]], axis=-1) # (H,W,2)
        return np.reshape(matrix, (-1, 2))
        
    def _compute_grid_size_matrix(self, grid_size, output_shape):
        h, w, _ = output_shape
        matrix = np.ones((h*w, 1), dtype='float32') * np.asarray(grid_size, dtype='float32')
        return matrix
        
    def compute_grid_base_matrix(self):
        matrix = np.concatenate(
            [ 
                self._compute_grid_base_matrix(output_shape) # (N,2)
                for output_shape in self.output_shapes 
            ], axis=0
        )
        matrix = np.tile(matrix, (self.n_batch, 1)) # (B*G, 2)
        return matrix
             
    def compute_grid_size_matrix(self):
        matrix = np.concatenate(
            [
                self._compute_grid_size_matrix(grid_size, output_shape) # (N,1)
                for grid_size, output_shape in zip(self.grid_sizes, self.output_shapes)
            ], axis=0
        )
        matrix = np.tile(matrix, (self.n_batch, 1)) # (B*G, 1)
        return matrix
             

class Decoder:
    def __init__(self, configs=None):
        # Hyper-parameters
        grid_sizes = np.asarray(configs['SHAPES_GRID'], dtype='float32') if 'SHAPES_GRID' in configs else np.array([
            [8,8],
            [16,16],
            [32,32],
        ], dtype='float32')

        output_shapes = np.asarray(configs['SHAPES_OUTPUT'], dtype='int32')
        self.input_shapes  = np.asarray(configs['SHAPES_INPUT'], dtype='float32')
        self.obj_threshold = np.asarray([configs['OBJ_THRESHOLD']], dtype='float32')
        self.iou_threshold = np.asarray([configs['NMS_THRESHOLD']], dtype='float32')
        self.pre_threshold = np.asarray([configs['PRE_THRESHOLD']], dtype='float32') if 'PRE_THRESHOLD' in configs else np.array([0.0], dtype='float32')

        self.n_box = configs['N_BOX_LOGIT'] if 'N_BOX_LOGIT' in configs else 4
        self.n_cls = configs['N_CLASSES']
        self.n_max = configs['N_MAX_OBJ']
        self.n_grid = (np.sum(np.prod(np.asarray(output_shapes, dtype='float32')[..., :-1], axis=-1))).astype('int32')

        self.refer = Encoder(output_shapes, grid_sizes, n_batch=self.n_max)
        self.grid_bases = self.refer.compute_grid_base_matrix() # (B_MAX*G, 2)
        self.grid_sizes = self.refer.compute_grid_size_matrix() # (B_MAX*G, 1)

    def decode_box(
        self, 
        box_pred, 
        scale_ratio, pre_bases, 
        grid_bases, grid_sizes,
    ):
        x1y1 = (grid_bases - box_pred[..., :2]) * grid_sizes
        x1y1 = x1y1 / scale_ratio + pre_bases
        x2y2 = (grid_bases + box_pred[..., 2:]) * grid_sizes
        x2y2 = x2y2 / scale_ratio + pre_bases

        boxes = np.concatenate([x1y1, x2y2], axis=-1)
        return boxes

    def get_scale_ratio(self, boxes):
        w = boxes[..., 2:3] - boxes[..., 0:1] + 1
        h = boxes[..., 3:4] - boxes[..., 1:2] + 1
        rw = self.input_shapes[1]
        rh = self.input_shapes[0]
        return np.concatenate([rw/w, rh/h], axis=-1) # (B,2)
        
    def split_logits(self, x, n, n_channels=(4, 80)):
        return np.reshape(x[:n * n_channels[0]], (-1, n_channels[0])), np.reshape(x[n * n_channels[0]:], (-1, n_channels[1]))

    def get_object_prob(self, x):
        x = sigmoid(x)
        idx = np.argmax(x, axis=-1)[..., None]
        prob = np.max(x, axis=-1)[..., None]
        return idx, prob

    def main(self, logits, reference):
        '''
        logits = ((N,4), (N,80))
        '''
        # Scaling Factor
        outputs = np.array([], dtype='float32')
        n_batch = len(reference)
        scale_ratio = np.repeat(self.get_scale_ratio(reference), self.n_grid, axis=0) # (B,2) -> (B*G,2)
        pre_bases = np.repeat(reference[..., :2], self.n_grid, axis=0)                # (B,2) -> (B*G,2)
        box_pred, cls_pred = self.split_logits(logits, self.n_grid)
        cls_idx, obj_pred = self.get_object_prob(cls_pred)

        mask = np.greater_equal(obj_pred[..., 0], max(self.pre_threshold, self.obj_threshold))
        if np.any(mask):
            obj_pred = obj_pred[mask]
            box_pred = box_pred[mask]
            cls_idx  = cls_idx[mask]
            scale_ratio = scale_ratio[mask]
            pre_bases = pre_bases[mask]
            grid_bases = self.grid_bases[:n_batch * self.n_grid, ...][mask]
            grid_sizes = self.grid_sizes[:n_batch * self.n_grid, ...][mask]
            
            # Decode
            box_pred = self.decode_box(
                box_pred,
                scale_ratio, pre_bases,
                grid_bases, grid_sizes,
            )

            # Masking NMS
            mask = py_nms(obj_pred, box_pred, self.iou_threshold)
            if np.any(mask):
                obj_pred = obj_pred[mask]
                box_pred = box_pred[mask]
                cls_idx  = cls_idx[mask]
                outputs = np.concatenate(
                    [obj_pred, cls_idx, box_pred], axis=-1
                )

        return outputs
