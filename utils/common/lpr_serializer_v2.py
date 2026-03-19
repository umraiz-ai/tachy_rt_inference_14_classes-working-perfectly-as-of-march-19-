#coding:utf-8

"""
Title: LPR Post-prossecing script (function of serialize)
Author: [Myungkyum Kim](dean@deeper-i.com)
"""


import os, sys
sys.path.append('libs')

from collections import OrderedDict as od
import json

import numpy as np
np.set_printoptions(threshold=np.inf)

from operations import convert_to_xywh, compute_distance


'''
Configuration
 - "BASE_CTG_FILE" : Category file

'''
class Serializer:
    '''
    Serializer
    1. Shapeing
        Converts the result to fit a specific format (country, type, ...)
    2. Stringify
        Positions -> strings 
    '''
    def __init__(self, cfg):
        assert 'BASE_CTG_FILE' in cfg.keys()
        self.ctg = self.get_ctg(cfg['BASE_CTG_FILE'])
        self.cfg = cfg
        self.shaping = Shaping(cfg)

    def get_ctg(self, ctg_f):
        if not ctg_f == None:
            with open(ctg_f, 'r') as j:
                ctg = json.load(j)
            return ctg
        else: return None

    def assign_by_location(self, ants_0, ants_1):
        '''
        Asign character to LP
        '''
        n = len(ants_0)
        boxes_0 = ants_0[..., 2:6]
        boxes_1 = ants_1[..., 2:6]
        center_0 = convert_to_xywh(boxes_0)[..., :2]    # (N,2)
        center_1 = convert_to_xywh(boxes_1)[..., :2]    # (M,2)
        dist = compute_distance(center_1, center_0)     # (M,N)
        cls = np.argmin(dist, axis=-1)[..., None]       # (M,1)
        return cls
    
    def stringify(self, elmts, categories=None):
        '''
        Convert class to category in order 
        '''
        line = ''
        for e in elmts: 
            i = e.item(0)
            if categories is not None:
                s = categories[i]
            else:
                s = str(i)
            line += s

        return np.array([line.encode('utf-8')], dtype='|S128')   

    
    def main(self, ants_0, ants_1):
        n = len(ants_0)
        strings = np.empty((n, 1), dtype='|S128')
        cbl = self.assign_by_location(ants_0, ants_1)
        for i in range(n):
            mask = cbl[..., 0] == i
            clss = self.shaping.matching(ants_1[mask])
            if len(clss) > 0:
                clss = self.shaping.align(clss)
                string = self.stringify(clss, categories=list(self.ctg.keys()) if self.ctg is not None else None)
            else:
                string = np.array([''], dtype='|S128')

            strings[i] = string

        return strings


##############################
### Shaping
##############################
class Shaping:
    def __init__(self, cfg):
        self.cfg = cfg

    def matching(self, ants):
        # TODO Matching format ...
        return ants

    def align(self, ants):
        str_list = []
        ind = np.argsort(ants[..., 2])
        ants_bx = ants[...][ind].astype('uint32')
        str_pivot = ants_bx[0][1]
        insert_idx = 0
        for i in range(len(ants_bx)):
            ymax = ants_bx[i][5]
            ymid = ((ants_bx[-1][5] - ants_bx[-1][3])/2) + ants_bx[-1][3]
                                                                          
            if ymax > ymid:
                str_list.append(ants_bx[i][1])
            else:
                str_list.insert(insert_idx, ants_bx[i][1])
                insert_idx += 1
            str_pivot = str_list[0]
        strings = np.array(str_list, dtype='uint32')
        return strings
