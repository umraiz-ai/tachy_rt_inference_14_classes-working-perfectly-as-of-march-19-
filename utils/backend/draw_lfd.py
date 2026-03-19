import cv2
import numpy as np

class draw_lfdae():
    def __init__(self, cfg):
        self.inf_size = cfg['ORG_SHAPE']
        self.dsp_size = cfg['DSP_SHAPE']

        self.prev_ants = np.array([])

        self.dis_bpm = 0
        self.non_out = 0
        self.cls_cnt = 0

        self.eye_info           = {}
        self.eye_info['sta']    = ''
        self.eye_info['clo']    = (0,0,255)
        self.eye_info['ahp']    = 0.0
        self.face_info          = {}
        self.face_info['clo']   = (206,206,206)
        self.agl_info           = {}
        self.agl_info['lclo']   = (206,206,206)
        self.agl_info['rclo']   = (206,206,206)
        self.agl_info['tclo']   = (206,206,206)
        self.agl_info['bclo']   = (206,206,206)

        self.bpm_info = {}
        self.bpm_info['sta'] = ''
        self.bpm_info['clo'] = {'NOM':(0,255,0), 'ABN':(0,0,255)}

        self.driver_status = ''

    def resize(self, x, interpolation=cv2.INTER_NEAREST):
        if self.size_resize is not None:
            x = cv2.resize(x, tuple(self.size_resize), interpolation=interpolation)
        return x

    def draw_box(self, img, pos, color=(206,206,206), thickness=3):
        pos = list(map(round, pos))
        pos = list(map(int, pos))
        img = cv2.rectangle(img.astype(np.uint8), (pos[0], pos[1]), (pos[2], pos[3]), color, thickness)
        return img

    def draw_ref_drt(self, img, box, xrfs, yrfs, color=(206,206,206), thickness=-1):
        xrfs = list(map(int, xrfs))
        yrfs = list(map(int, yrfs))
        img = cv2.rectangle(img.astype(np.uint8), (int(box[2]+(box[2]*(1/20))), yrfs[0]), (int(box[2]+(box[2]*(1/20)+20)), yrfs[1]), self.agl_info['lclo'], thickness) # right ref
        img = cv2.rectangle(img.astype(np.uint8), (int(box[0]-(box[2]*(1/20))), yrfs[0]), (int(box[0]-(box[2]*(1/20)+20)), yrfs[1]), self.agl_info['rclo'], thickness) # left ref

        img = cv2.rectangle(img.astype(np.uint8), (xrfs[0], (int(box[1]-(box[3]*(1/15)+20)))), (xrfs[1], (int(box[1]-(box[3]*(1/15))))), self.agl_info['tclo'], thickness) # top
        img = cv2.rectangle(img.astype(np.uint8), (xrfs[0], (int(box[3]+(box[3]*(1/15))))), (xrfs[1], (int(box[3]+(box[3]*(1/15)+20)))), self.agl_info['bclo'], thickness) # bottom

        return img

    def draw_circle(self, img, pos, radian=5, color=(206,206,206), thickness=-1):
        back_g = img.copy()
        pos = list(map(round, pos))
        pos = list(map(int, pos))
        back_g = cv2.circle(back_g, (pos[0], pos[1]), radian, color, thickness)
                                                                              
        img = cv2.addWeighted(img, self.eye_info['ahp'], back_g, 1-self.eye_info['ahp'], 0)
        
        return img

    def draw_linebyangle(self, img, cen, agl, color=(0,255,0), thickness=10):
        yaw, pit, roll = agl
        yaw_cords  = [int(cen[0]-yaw*7), cen[1]]
        pit_cords  = [cen[0], int(cen[1]-pit*7)]
        roll_cords = [int(cen[0]+roll*7), int(cen[1]+roll*7)]
        # img = cv2.line(img, tuple(cen), tuple(yaw_cords*2), (255,0,0), thickness)
        # img = cv2.line(img, tuple(cen), tuple(pit_cords*2), (0,255,0), thickness)
        # img = cv2.line(img, tuple(cen), tuple(roll_cords*2), (0,0,255), thickness)
        img = cv2.line(img, tuple(cen), tuple(yaw_cords), (255,0,0), thickness)
        img = cv2.line(img, tuple(cen), tuple(pit_cords), (0,255,0), thickness)
        img = cv2.line(img, tuple(cen), tuple(roll_cords), (0,0,255), thickness)

        return img

    def put_txt(self, img, txt, x, y, scale=1.0, color=(0,0,255), t=2):
        img = cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)

        return img

    def add_img_blend(self, img, bld):
        img = cv2.addWeighted(img, 0.5, bld, 0.5, 0)
        
        return img

    def decide_e_sta(self, sta):
        pred_close = sta[1]
        pred_close = pred_close
        self.eye_info['ahp'] = pred_close

    def decide_b_sta(self, bpm):
        if bpm > 0: self.bpm_info['sta'] = 'NOM'
        else: self.bpm_info['sta'] = 'ABN'
        
    def decide_direction(self, yaw, pit):
        ang_cls = 'CEN'
        if -7.5 <= pit < 7.5 and -12.5 <= yaw < 12.5: 
            ang_cls = 'CEN'
            self.agl_info['lclo']   = (206,206,206)
            self.agl_info['rclo']   = (206,206,206)
            self.agl_info['tclo']   = (206,206,206)
            self.agl_info['bclo']   = (206,206,206)
        elif -7.5 <= pit < 7.5 and -20.5 >= yaw: 
            lft_ahp = abs(int(round(yaw*3)))
            ang_cls = 'LFT'
            self.agl_info['lclo']   = (175-lft_ahp,175-lft_ahp,255)
            self.agl_info['rclo']   = (206,206,206)
            self.agl_info['tclo']   = (206,206,206)
            self.agl_info['bclo']   = (206,206,206)
        elif -7.5 <= pit < 7.5 and 20.5 <= yaw: 
            rgt_ahp = abs(int(round(yaw*2)))
            ang_cls = 'RGT'
            self.agl_info['rclo']   = (175-rgt_ahp,175-rgt_ahp,255)
            self.agl_info['lclo']   = (206,206,206)
            self.agl_info['tclo']   = (206,206,206)
            self.agl_info['bclo']   = (206,206,206)
        elif pit > 12.5  and -12.5 <= yaw < 12.5: 
            top_ahp = abs(int(round(pit*2)))
            ang_cls = 'TOP'
            self.agl_info['tclo']   = (175-top_ahp,175-top_ahp,255)
            self.agl_info['lclo']   = (206,206,206)
            self.agl_info['rclo']   = (206,206,206)
            self.agl_info['bclo']   = (206,206,206)
        elif pit < -12.5 and -12.5 <= yaw < 12.5: 
            btm_ahp = abs(int(round(pit*2)))
            ang_cls = 'BTM'
            self.agl_info['bclo']   = (175-btm_ahp,175-btm_ahp,255)
            self.agl_info['lclo']   = (206,206,206)
            self.agl_info['rclo']   = (206,206,206)
            self.agl_info['tclo']   = (206,206,206)
        else: ang_cls = 'ABN'
                                                                              
        return ang_cls

    def refine_box(self, box):
        # TODO box mean per frames
        return box

    def get_drt_refcords(self, box):
        h = box[3] - box[1]
        w = box[2] - box[0]

        y_ref1 = box[3]-(h*(3/4))
        y_ref2 = box[3]-(h*(1/4))

        x_ref1 = box[2]-(w*(3/4))
        x_ref2 = box[2]-(w*(1/4))

        return x_ref1, x_ref2, y_ref1, y_ref2

    def main(self, x):
        img    = x[0] 
        ants_0 = x[1] # (N,16) 

        if len(ants_0) == 0: 
            ants_0 = self.prev_ants
            self.non_out += 1

        if self.non_out > 50: 
            ants_0 = np.array([])
            self.non_out = 0

        ratios = [(self.inf_size[0] / self.dsp_size[0]), (self.inf_size[1] / self.dsp_size[1])]


        bld = np.zeros(img.shape, dtype=np.uint8)
        bld[...,2] = 255

        if len(ants_0) > 0:
            idx = np.argmax(np.exp(ants_0[...,0:1]))
            stc = ants_0[idx][1]
            box  = ants_0[idx][2:6]
            trg_box = self.refine_box(box)
            trg_box = trg_box / np.array([ratios[0], ratios[1], ratios[0], ratios[1]])
            cen_3line = [int((trg_box[2] + trg_box[0])/2),int(trg_box[1])]
            xrf1, xrf2, yrf1, yrf2 = self.get_drt_refcords(trg_box)
            sta  = ants_0[idx][13:16]
            le   = ants_0[idx][6:8] / np.array([ratios[0], ratios[1]])
            re   = ants_0[idx][8:10] / np.array([ratios[0], ratios[1]])
            agl  = ants_0[idx][10:13] * 180. / np.pi
            face_drt = self.decide_direction(agl[0], agl[1])
            self.decide_e_sta(sta)

            img = self.draw_box(img, trg_box, color=self.face_info['clo'])
            img = self.draw_circle(img, le, color=self.eye_info['clo'])
            img = self.draw_circle(img, re, color=self.eye_info['clo'])
            img = self.put_txt(img, self.eye_info['sta'], 100, 10, color=self.eye_info['clo'])
            img = self.draw_ref_drt(img, trg_box, (xrf1,xrf2), (yrf1, yrf2))
            img = self.draw_linebyangle(img, cen_3line, agl)

            if stc == 1.0: 
                self.cls_cnt += 0.5
            elif stc == 0.0: self.cls_cnt -= 1.5
            if self.cls_cnt < 0: self.cls_cnt = 0
            
            if agl[1] < -20 and self.cls_cnt > 5: self.driver_status = 'DROWSE'
            elif self.cls_cnt > 15 : self.driver_status = 'DROWSE'
            elif agl[0] > 22.5 or agl[0] < -22.5 or agl[1] < -15: self.driver_status = 'DROWSE'
            # elif agl[0] > 22.5 or agl[0] < -22.5 or agl[1] < -15: self.driver_status = 'CAUTION'
            else: self.driver_status = 'NOMAL'
            
            if self.driver_status == 'NOMAL': 
                img = self.put_txt(img, "{}".format(self.driver_status), int(trg_box[2]-(trg_box[2]-trg_box[0])/2)-100, int(trg_box[3]-30), scale=2, color=(0,255,0), t=4)
            else:
                img = self.put_txt(img, "{}".format(self.driver_status), int(trg_box[2]-(trg_box[2]-trg_box[0])/2)-100, int(trg_box[3]-30), scale=2, color=(0,0,255), t=4)

        self.prev_ants = ants_0

        return img
