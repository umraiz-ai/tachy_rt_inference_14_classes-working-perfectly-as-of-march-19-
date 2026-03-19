#coding:utf-8

"""
Title: Character Detection in LPR Application, Postprocess and Methods(Encoder, Decoder)
Author: [Myungkyum Kim](dean@deeper-i.com)
Model: character82_detection_yolov40
Description: Utilities for Yolov4 BBox, Class
History:
    2022/05/06: init
"""


import os, sys
import time

import numpy as np

from operations import sigmoid, convert_to_corners, py_nms

# class post_processor:
#     def __init__(self, dynamic_config=None):
#         self.anchors = \
#         [
#             [
#                 [10,14],[23,27],[37,58]
#             ],
#             [
#                 [81,82],[135,169],[344,319]
#             ]
#         ]
#         self.shapes_grid = [[16,16],[32,32]]
# 
#         self.shapes_input = dynamic_config["SHAPES_INPUT"]
#         self.shapes_output = dynamic_config["SHAPES_OUTPUT"]
#         self.pre_thres = dynamic_config["PRE_THRESHOLD"] if "PRE_THRESHOLD" in dynamic_config else 0.1
#         self.obj_thres = dynamic_config["OBJ_THRESHOLD"] if "OBJ_THRESHOLD" in dynamic_config else 0.3
#         self.nms_thres = dynamic_config["NMS_THRESHOLD"] if "NMS_THRESHOLD" in dynamic_config else 0.3
#         self.n_max_obj = dynamic_config["N_MAX_OBJ"] if "N_MAX_OBJ" in dynamic_config else 20
# 
#         config = self._merge_config()
#         self.decoder = Decoder(config)
# 
#     def set_pre_thres(self, value):
#         if 0.0 <= value <= 1.0:
#             self.pre_thres = value
#             self.decoder.pre_threshold = value
# 
#     def set_obj_thres(self, value):
#         if 0.0 <= value <= 1.0:
#             self.obj_thres = value
#             self.decoder.obj_threshold = value
# 
#     def set_nms_thres(self, value):
#         if 0.0 <= value <= 1.0:
#             self.nms_thres = value
#             self.decoder.nms_threshold = value
# 
#     def main(self, logits, references):
#         logits = logits.reshape(-1,8)
#         t = time.time()
#         ret = self.decoder.main(logits, references)
#         print("od post time :", time.time() - t)
#         return ret
# 
#     def _merge_config(self):
#         config = {
#             'ANCHORS' : self.anchors,
#             'SHAPES_GRID' : self.shapes_grid,
#             'SHAPES_INPUT' : self.shapes_input,
#             'SHAPES_OUTPUT' : self.shapes_output,
# 
#             'PRE_THRESHOLD' : self.pre_thres,
#             'OBJ_THRESHOLD' : self.obj_thres,
#             'NMS_THRESHOLD' : self.nms_thres,
# 
#             'N_OBJ_LOGIT' : 1,
#             'N_BOX_LOGIT' : 4,
#             'N_CLASSES'   : 3,
#             'N_MAX_OBJ'   : self.n_max_obj
#         }
#         return config

#############
#### Methods
#############
class Encoder:
    def __init__(self, output_shapes, grid_sizes, anchor_sizes, n_batch=10):
        self.anchor_sizes = anchor_sizes
        self.output_shapes = output_shapes
        self.grid_sizes = grid_sizes
        self.n_batch = n_batch

    def _compute_grid_base_matrix(self, output_shape, n_anchor):
        h, w, _ = output_shape
        if h == w:
            matrix = np.stack(np.meshgrid(range(h), range(w)), axis=-1) 
            matrix = np.tile(matrix[:, :, None, :], (1, 1, n_anchor, 1)).astype('float32')
        else:
            x = np.tile(np.array(range(w)), (h,1))
            y = np.tile(np.array(range(h)), (w,1)).T # (H,W)
            matrix = np.tile(np.array([x, y], dtype='float32'), (n_anchor,1,1,1)).transpose((2,3,0,1)) # (H,W,A,2)
        return np.reshape(matrix, (-1, 2))
        
    def _compute_grid_size_matrix(self, grid_size, output_shape, n_anchor):
        h, w, _ = output_shape
        matrix = np.ones((h*w*n_anchor, 1), dtype='float32') * np.asarray(grid_size, dtype='float32')
        return matrix
        
    def _compute_anchor_matrix(self, output_shape, anchors):
        h, w, _ = output_shape
        matrix = np.tile(np.asarray(anchors, dtype='float32')[None, None, ...], (h, w, 1, 1)) # (H,W,A,2)
        return np.reshape(matrix, (-1, 2))
        

    def compute_grid_base_matrix(self):
        matrix = np.concatenate(
            [ 
                self._compute_grid_base_matrix(output_shape, len(anchor)) # (N,2)
                for output_shape, anchor in zip(self.output_shapes, self.anchor_sizes) 
            ], axis=0
        )
        matrix = np.tile(matrix, (self.n_batch, 1)) # (B*G, 2)
        return matrix
             
    def compute_grid_size_matrix(self):
        matrix = np.concatenate(
            [
                self._compute_grid_size_matrix(grid_size, output_shape, len(anchor)) # (N,1)
                for grid_size, output_shape, anchor in zip(self.grid_sizes, self.output_shapes, self.anchor_sizes)
            ], axis=0
        )
        matrix = np.tile(matrix, (self.n_batch, 1)) # (B*G, 1)
        return matrix
             
    def compute_anchor_matrix(self):
        matrix = np.concatenate(
            [
                self._compute_anchor_matrix(output_shape, anchor_size) # (N,2)
                for output_shape, anchor_size in zip(self.output_shapes, self.anchor_sizes)
            ], axis=0
        )
        matrix = np.tile(matrix, (self.n_batch, 1)) # (B*G, 2)
        return matrix
             

class Decoder:
    def __init__(self, configs=None):
        # Hyper-parameters
        anchor_sizes = np.asarray(configs['ANCHORS'], dtype='float32')
        grid_sizes = np.asarray(configs['SHAPES_GRID'], dtype='float32')
        output_shapes = np.asarray(configs['SHAPES_OUTPUT'], dtype='int32')

        self.n_grid = (np.sum(np.prod(np.asarray(output_shapes, dtype='float32')[..., :-1], axis=-1)) * anchor_sizes.shape[1]).astype('int32')
        self.input_shapes  = np.asarray(configs['SHAPES_INPUT'], dtype='float32')
        self.pre_threshold = np.asarray([configs['PRE_THRESHOLD']], dtype='float32') if 'PRE_THRESHOLD' in configs else np.array([0.0], dtype='float32')
        self.obj_threshold = np.asarray([configs['OBJ_THRESHOLD']], dtype='float32')
        self.iou_threshold = np.asarray([configs['NMS_THRESHOLD']], dtype='float32')

        self.n_obj = configs['N_OBJ_LOGIT'] if 'N_OBJ_LOGIT' in configs else 1
        self.n_box = configs['N_BOX_LOGIT'] if 'N_BOX_LOGIT' in configs else 4
        self.n_cls = configs['N_CLASSES']
        self.n_max = configs['N_MAX_OBJ']

        self.refer = Encoder(output_shapes, grid_sizes, anchor_sizes, n_batch=self.n_max)
        self.grid_bases = self.refer.compute_grid_base_matrix() # (B_MAX*G, 2)
        self.grid_sizes = self.refer.compute_grid_size_matrix() # (B_MAX*G, 1)
        self.anchors = self.refer.compute_anchor_matrix()       # (B_MAX*G, 2)

    def decode_box(
        self, 
        box_pred, 
        scale_ratio, pre_bases, 
        grid_bases, grid_sizes, anchors,
    ):
        boxes = np.concatenate(
            [
                ((box_pred[..., :2] * 2. - 0.5 + grid_bases) * grid_sizes) / scale_ratio + pre_bases,
                (box_pred[..., 2:] * 2.) ** 2 * anchors / scale_ratio,
            ], axis=-1
        )
        boxes = convert_to_corners(boxes)

        return boxes

    def get_scale_ratio(self, boxes):
        w = boxes[..., 2:3] - boxes[..., 0:1] + 1
        h = boxes[..., 3:4] - boxes[..., 1:2] + 1
        rw = self.input_shapes[1]
        rh = self.input_shapes[0]
        return np.concatenate([rw/w, rh/h], axis=-1) # (B,2)
        
    def main(self, logits, reference):
        '''
        logits = (B*H*W*A, 1+2+2+C)
        '''
        # Scaling Factor
        logits = logits.reshape(-1,8)
        outputs = np.array([], dtype='float32')
        n_batch = len(reference)
        scale_ratio = np.repeat(self.get_scale_ratio(reference), self.n_grid, axis=0) # (B,2) -> (B*G,2)
        pre_bases = np.repeat(reference[..., :2], self.n_grid, axis=0)                # (B,2) -> (B*G,2)
        obj_pred = sigmoid(logits[..., 0:self.n_obj])
        mask = np.greater_equal(obj_pred[..., 0], self.pre_threshold)
        if np.any(mask):
            logit = logits[..., 1:][mask]
            obj_pred = obj_pred[mask]; offset = 0
            box_pred = sigmoid(logit[..., offset:offset + self.n_box]); offset += self.n_box
            cls_pred = sigmoid(logit[..., offset:offset + self.n_cls]); offset += self.n_cls
            scale_ratio = scale_ratio[mask]
            pre_bases = pre_bases[mask]
            grid_bases = self.grid_bases[:n_batch * self.n_grid, ...][mask]
            grid_sizes = self.grid_sizes[:n_batch * self.n_grid, ...][mask]
            anchors = self.anchors[:n_batch * self.n_grid, ...][mask]
            
            # Masking threshold 
            mask = np.greater_equal(
                obj_pred[..., 0] * np.max(cls_pred, axis=-1), 
                self.obj_threshold
            )
            if np.any(mask):
                obj_pred = obj_pred[mask]
                box_pred = box_pred[mask]
                cls_pred = cls_pred[mask]

                scale_ratio = scale_ratio[mask]
                pre_bases = pre_bases[mask]
                grid_bases = grid_bases[mask]
                grid_sizes = grid_sizes[mask]
                anchors = anchors[mask]

                # Decode
                box_pred = self.decode_box(
                    box_pred, 
                    scale_ratio, pre_bases,
                    grid_bases, grid_sizes, anchors,
                )
                # Masking NMS 

                mask = py_nms(obj_pred, box_pred, self.iou_threshold)
                # idx_nms = nms_v2(obj_pred, box_pred, self.iou_threshold)
                if np.any(mask):
                    obj_pred = obj_pred[mask]
                    box_pred = box_pred[mask]
                    cls_pred = np.argmax(cls_pred[mask], axis=-1)[..., None]
                    outputs = np.concatenate(
                        [obj_pred, cls_pred, box_pred], axis=-1
                    )
        
        return outputs
