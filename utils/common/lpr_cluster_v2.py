#coding:utf-8


import numpy as np


'''
Configuration
    CST_BYPASS     : false
    CST_N_KEEP     : 2,
    CST_N_PICK     : 10,
    CST_PART_MATCH : true,
    CST_PERIOD     : 200
'''
class Cluster:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.period = cfg['CST_PERIOD'] if 'CST_PERIOD' in cfg.keys() else 0
        self.n_pick = cfg['CST_N_PICK'] if 'CST_N_PICK' in cfg.keys() else 1
        self.n_keep = cfg['CST_N_KEEP'] if 'CST_N_KEEP' in cfg.keys() else 1
        self.bypass = cfg['CST_BYPASS'] if 'CST_BYPASS' in cfg.keys() else False
        self.pm = cfg['CST_PART_MATCH'] if 'CST_PART_MATCH' in cfg.keys() else False

        self.db_cnt = {}
        self.db_gap = {}
        self.block_list = []
        self.prev_block_list = []
        self.cnt = 0
        # self.block_str = b'FFFFFFFF'
        self.block_str = b''

        self.hangle_list = [
                            '가', '나', '다', '라', '마', '바', '사', '아', '자', '하', 
                            '거', '너', '더', '러', '머', '버', '서', '어', '저', '허', 
                            '고', '노', '도', '로', '모', '보', '소', '오', '조', '호', 
                            '구', '누', '두', '루', '무', '부', '수', '우', '주', '배', 
                          ]

    ####################################
    ### Algorithm: similarity by part for KR-LP
    ####################################
    # A method of dividing a character string into 3 parts and determining the same character string when two or more are the same
    def _segmentation_lp_kr(self, string):
        string = string.decode('utf-8')
        n = len(string)
        t = -1
        for i, c in enumerate(string[::-1]):
            if c in self.hangle_list:
                t = n - 1 - i
        if t == -1:
            return string, string, string
        else:
            return [string[0:t], string[t:t+1], string[t+1:]]

    def _match(self, trg, db):
        cnt = 0
        for t, d in zip(trg, db):
            if t == d: cnt += 1
        return cnt
            
    def _blocking_by_similarity(self, picks):
        blocks = []
        for s in picks:
            trg = self._segmentation_lp_kr(s)
            for ss in self.block_list:
                db = self._segmentation_lp_kr(ss)
                if self._match(trg, db) >= 2:
                    blocks.append(s)
                    break

        picks = list(set(picks) - set(blocks))
        self.block_list += blocks

        return picks

    def _increase_gap_all(self):
        for s in self.db_gap:
            self.db_gap[s] += 1

        return self
        
    def _decrease_gap(self, string):
        if string in self.db_gap:
            self.db_gap[string] -= 1

        return self

    def _clear_block_list(self):
        if self.cnt >= self.period:
            self.block_list = list(set(self.block_list) - set(self.prev_block_list))
            self.prev_block_list = self.block_list
            self.cnt = 0
        else:
            self.cnt += 1

        return self

    def _update(self, string):
        if string == '': 
            return self

        if string in self.db_cnt.keys():
            self.db_cnt[string] += 1
            self.db_gap[string] = 0
        else:
            self.db_cnt[string] = 1
            self.db_gap[string] = 0

        return self

    def _flush_by_gap(self):
        del_list = []
        for s in self.db_gap:
            if self.db_gap[s] >= self.n_keep + 1:
                del_list.append(s)

        for s in del_list:
            del self.db_gap[s]
            del self.db_cnt[s]

        return self

    def pick(self):
        pick_list = []
        for s in self.db_cnt:
            if self.db_cnt[s] >= self.n_pick: 
                pick_list.append(s)

        for s in pick_list:
            del self.db_cnt[s]
            del self.db_gap[s]
        
        # Refresh block list by similarity 
        # pick_list = list(set(pick_list) - set(self.block_list + [self.block_str]))
        pick_list = list(set(pick_list) - set(self.block_list))
        return pick_list



    def main(self, strings):
        if self.bypass: return strings, True

        strings = strings[(strings != self.block_str)]
        self._increase_gap_all()
        for s in strings:
            s = s.item(0)
            self._decrease_gap(s)
            self._update(s)
            
        p = self.pick()
        if self.pm: p = self._blocking_by_similarity(p)
        self.block_list += p
        if len(p) > 0:
            valid = True
            outputs = np.array(p, dtype='|S128')[..., None]
        else:
            valid = False
            outputs = np.array([], dtype='|S128')

        # Update by gap
        self._flush_by_gap()

        # Clear block list
        self._clear_block_list()

        return outputs, valid




