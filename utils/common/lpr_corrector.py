#coding:utf-8

"""
Title: LPR Post-prossecing script (function of serialize)
Author: [Myungkyum Kim](dean@deeper-i.com)
"""


from collections import OrderedDict as od
import numpy as np
np.set_printoptions(threshold=np.inf)


class Corrector:
    '''
    Serializer
    1. Filtering
        Detected characters are converted into a single string with license plate format
    2. Voting
        Regenerate the results sorted based on location using information from the previous frame    
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.vot = VotingByValue(cfg)
        self.filtering = Filtering(cfg)

    def main(self, strings):
        strings = self.filtering.main(strings)
        strings = self.vot.main(strings)
        return strings


##############################
### Voting
##############################
class VotingByValue:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db = od({}) # {0:[['ABC1234', 'ABC1233'], 0], 1:[], ...}
        self.cur_id = 0
        self.n_buf = int(cfg['VOT_N_BUF']) if 'VOT_N_BUF' in cfg.keys() else 5
        self.min_val = int(cfg['VOT_MIN_VAL']) if 'VOT_MIN_VAL' in cfg.keys() else 4
        self.period = int(cfg['VOT_PERIOD']) if 'VOT_PERIOD' in cfg.keys() else 30
        self.bypass = cfg['VOT_BYPASS'] if 'VOT_BYPASS' in cfg.keys() else True

    def _initialize(self, _id):
        if _id not in self.db.keys():
            self.db[_id] = [[], 0]
        return self

    def _increase(self):
        for _id in self.db:
            self.db[_id][1] += 1
        return self

    def _flush(self):
        result = []
        for _id in self.db:
            if self.db[_id][1] > self.period:
                result.append(_id)

        for _id in result:
            del self.db[_id]
                
        return self

    def _update(self, _id, val):
        if len(self.db[_id][0]) < self.n_buf:
            self.db[_id][0].append(val)
        else:
            self.db[_id][0][0:self.n_buf-1] = self.db[_id][0][1:self.n_buf]
            self.db[_id][0][self.n_buf-1] = val

        self.db[_id][1] = 0
        self.cur_id += 1
        return self

    def _compare(self, ref, val):
        if len(ref) > len(val):
            for i in range(len(val) - self.min_val + 1):
                if val[i:i+self.min_val] in ref:
                    return True
        else:
            for i in range(len(ref) - self.min_val + 1):
                if ref[i:i+self.min_val] in val:
                    return True
        return False

    def _select(self, candidates, sel_id, sel_val):
        '''
        Most large amount and long string
        '''
        n_max = 0
        key = sel_id
        val = sel_val
        for _id in candidates:
            if len(self.db[_id][0]) > n_max:
                n_max = len(self.db[_id][0])
                key = _id 
                val = self.db[_id][0][-1] if len(self.db[_id][0][-1]) > len(val) else val

        return key, val
            
    def pick(self, string):
        sel_id = self.cur_id
        sel_val = string
        candidates = []
        for _id in self.db:
            for s in self.db[_id][0]:
                if self._compare(s, string):
                    candidates.append(_id)
                    break

        sel_id, sel_val = self._select(candidates, sel_id, sel_val)
        self._initialize(sel_id)        
        self._update(sel_id, sel_val)
        self._increase()
        self._flush()
        return sel_id, sel_val

    def main(self, strings, info=None):
        if self.bypass: return strings
        if len(strings) > 0:
            result = []
            for s in strings:
                ss = s.item(0)
                if str(ss.decode('utf-8')) != '': 
                    _, ss = self.pick(ss)
                    result.append(ss)

            strings_new = np.array(result, dtype='|S128')

        else:
            strings_new = strings
        return strings_new


##############################
### Filtering
##############################
class Filtering:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bypass = cfg['FLT_BYPASS'] if 'FLT_BYPASS' in cfg.keys() else False
        self.k_lp_chars = [
            '가', '나', '다', '라', '마', '바', '사', '아', '자', '하', 
            '거', '너', '더', '러', '머', '버', '서', '어', '저', '허', 
            '고', '노', '도', '로', '모', '보', '소', '오', '조', '호', 
            '구', '누', '두', '루', '무', '부', '수', '우', '주', '배', 
        ]
        self.k_lp_regions = [
            '서울', '대구', '광주', '울산', '경기', '충북', '전북',
            '경북', '제주', '부산', '인천', '대전', '세종', '강원',
            '충남', '전남', '경남'
        ]

    def check_line_count(self, line):
        # if 7 <= len(line) <= 9:
        #     return line
        # else: return None
        return line if 7 <= len(line) <= 9 else None

    def check_pur_delimiter(self, line):
        if line is not None:
            for i in range(2, len(line)):
                if line[i] in self.k_lp_chars:
                    self.sc_idx = i
                    self.sc_flag = True
                    break

            if self.sc_flag: return line
            else: return None

        else: return None

    def check_all_num(self, s_line):
        try: 
            for i in range(len(s_line)):
                s_int = int(s_line[i])
        except ValueError: return False
        
        return True

    def check_contain_region(self, s_line):
        if s_line[:2] in self.k_lp_regions: return True
        else: return False

    def check_after_dm(self, line):
        if line is not None:
            back_chars = line[self.sc_idx+1:]
            if len(back_chars) == 4: 
                if self.check_all_num(back_chars): return line
                else: return None
            else: return None

        else: return None

    def check_before_dm(self, line):
        if line is not None:
            front_chars = line[:self.sc_idx]
            if len(front_chars) == 2:
                if self.check_all_num(front_chars): return line
                else: return None
            elif len(front_chars) == 3:
                if self.check_contain_region(front_chars): 
                    if self.check_all_num(front_chars[2]): return line
                    else: return None
                else: 
                    if self.check_all_num(front_chars): return line
                    else: return None
                return line
            elif len(front_chars) == 4:
                if self.check_contain_region(front_chars): 
                    if self.check_all_num(front_chars[2:4]): return line
                    else: return None
                else: return None

        else: return None

    # def main(self, line):
    #     if self.bypass: return line
    #     self.sc_flag = False
    #     self.sc_idx = 0

    #     line = self.check_line_count(line) # check Korea LP ocr length
    #     line = self.check_pur_delimiter(line) # check that Korea LP ocr contains the purpose delimiter
    #     line = self.check_after_dm(line) # check 4 number after the usage delimiter
    #     line = self.check_before_dm(line)

    #     if line is None:
    #         # return str(random.randrange(0,10000))
    #         return np.array(['FFFFFFFF'], dtype='|S128')
    #     else:
    #         return np.array([line.encode('utf-8')], dtype='|S128')   

    #     # return line

    def main(self, lines):
        if self.bypass: return lines
        strings = np.empty((len(lines), 1), dtype='|S128')
        for i, line in enumerate(lines):
            self.sc_flag = False
            self.sc_idx = 0

            line = line.item(0).decode('utf-8')
            line = self.check_line_count(line) # check Korea LP ocr length
            line = self.check_pur_delimiter(line) # check that Korea LP ocr contains the purpose delimiter
            line = self.check_after_dm(line) # check 4 number after the usage delimiter
            line = self.check_before_dm(line)

            if line is None:
                line = np.array([''], dtype='|S128')
            else:
                line = np.array([line.encode('utf-8')], dtype='|S128')   
            strings[i] = line

        return strings

