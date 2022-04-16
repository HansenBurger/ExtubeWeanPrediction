import sys
from pathlib import Path
from tkinter.tix import Tree

sys.path.append(str(Path.cwd()))

from operator import add, mul
from WaveDataProcess import BinImport
from Classes.Domain import layer_0, layer_1, layer_2
from Classes.Func.KitTools import PathVerify, LocatSimiTerms
from Classes.Func.CalculatePart import IndCalculation, VarAnalysis


class Basic():
    def __init__(self) -> None:
        pass


class RecordResp(Basic):
    def __init__(self, parent, id_):
        super().__init__()
        self.__rec = layer_1.RecWave()
        self.__rec.zdt = PathVerify(parent) / (id_ + '.zdt')

    @property
    def rec(self):
        return self.__rec

    def WaveformInit(self):
        rec = self.__rec
        p_wave = BinImport.WaveData(self.__rec.zdt)
        wave = p_wave.WaveDataGet()
        try:
            rec.sr = p_wave.resr
            rec.p_ind = wave['ind']
            rec.s_ind = wave['ind'][:len(wave['ind']) - 1]
            rec.e_ind = [x - 1 for x in wave['ind'][1:]]
            rec.s_F, rec.s_P, rec.s_V = wave['s_F'], wave['s_P'], wave['s_V']
        except:
            rec = None

    def IndicatorCalculate(self):
        rec = self.__rec
        if not rec:
            return

        for i in range(len(rec.p_ind) - 1):
            resp = layer_0.Resp()
            p_count = IndCalculation(rec.s_ind[i], rec.e_ind[i])
            p_count.ValidityCheck(rec.s_F, rec.s_V, rec.s_P)

            if not p_count.valid_tag:
                resp.val = False
            else:
                resp.wid = p_count.RespT(rec.sr)
                resp.rr, rr_val = p_count.RR(rec.sr)

                v_t, v_t_val = p_count.V_t()
                resp.v_t_i = v_t['v_t_i']
                resp.v_t_e = v_t['v_t_e']

                wob, wob_val = p_count.WOB()
                resp.wob = wob['wob']
                resp.wob_f = wob['wob_f']
                resp.wob_a = wob['wob_a']
                resp.wob_b = wob['wob_b']

                resp.ve, ve_val = p_count.VE(resp.rr, resp.v_t_i)
                resp.rsbi = p_count.RSBI(resp.rr, resp.v_t_i)

                mp_d, mp_d_val = p_count.MP_Area(resp.rr, resp.v_t_i, resp.wob)
                resp.mp_jm_d = mp_d['mp_jm']
                resp.mp_jl_d = mp_d['mp_jl']

                mp_t, mp_t_val = p_count.MP_Area(resp.rr, resp.v_t_i,
                                                 resp.wob_f)
                resp.mp_jm_t = mp_t['mp_jm']
                resp.mp_jl_t = mp_t['mp_jl']

                val_list = [
                    rr_val, v_t_val, wob_val, ve_val, mp_d_val, mp_t_val
                ]
                resp.val = True if not False in val_list else False

            rec.resps.append(resp)


class RecordPara(Basic):
    def __init__(self, parent, id_):
        super().__init__()
        self.__rec = layer_1.RecPara()
        self.__rec.zpx = PathVerify(parent) / (id_ + '.zpx')

    @property
    def rec(self):
        return self.__rec

    def __FieldVal(self, para, key_):
        try:
            value = para[key_]
        except:
            value = None
        return value

    def ParametersInit(self, machine):
        rec = self.__rec
        p_ = BinImport.ParaData(self.__rec.zpx)
        para = p_.ParaInfoGet()
        if not para:
            rec = None
        else:
            rec.u_ind = self.__FieldVal(para, 'uiDataIndex')
            rec.st_peep = self.__FieldVal(para, 'st_PEEP')
            rec.st_ps = self.__FieldVal(para, 'st_P_SUPP')
            rec.st_e_sens = self.__FieldVal(para, 'st_E_SENS')
            rec.bed_hr = self.__FieldVal(para, 'bed_HR')
            rec.bed_sbp = self.__FieldVal(para, 'bed_SBP')
            rec.bed_dbp = self.__FieldVal(para, 'bed_DBP')
            rec.bed_mbp = self.__FieldVal(para, 'bed_MBP')
            rec.bed_spo2 = self.__FieldVal(para, 'bed_SpO2')
            rec.bed_rr = self.__FieldVal(para, 'bed_RR')
            rec.bed_pr = self.__FieldVal(para, 'bed_PR')
            rec.bed_cvpm = self.__FieldVal(para, 'bed_CVPm')
            rec.st_sump = list(map(add, rec.st_peep, rec.st_ps))

            try:
                rec.st_mode = p_.VMInter(machine, slice(0, len(rec.u_ind)))
            except:
                rec.st_mode = []

    def ParaSelectBT(self, SR, time_tag, para_l):
        rec = self.__rec
        t_ind = list(map(mul, rec.u_ind, [1 / SR] * len(rec.u_ind)))
        s_ind = LocatSimiTerms(t_ind, time_tag).values()
        p_s_l = [para_l[i] for i in s_ind if i != None]
        return p_s_l


class RecordInfo(Basic):
    def __init__(self, parent, t_, id_):
        super().__init__()
        self.__rec = layer_2.RidRec()
        self.__rec.zif = PathVerify(parent) / (
            str(t_.year) + str(t_.month).rjust(2, '0')) / id_ / (id_ + '.zif')

    @property
    def rec(self):
        return self.__rec

    def ParametersInit(self):
        rec = self.__rec
        p_ = BinImport.RidData(self.__rec.zif)
        info = p_.RecordInfoGet()
        recs = p_.RecordListGet()
        try:
            rec.vm_n = info['m_n']
            rec.cls_t = max(recs['s_t'])
        except:
            print('zif file error')
            rec = None


class ResultStatistical(Basic):
    def __init__(self, resp_list):
        super().__init__()
        self.__rec = layer_2.Result()
        self.__resp_l = resp_list
        self.__rec.td = layer_1.DomainTS()
        self.__rec.fd = layer_1.DomainFS()
        self.__rec.hra = layer_1.DomainHRA()
        self.__rec.hrv = layer_1.DomainHRV()

    @property
    def rec(self):
        return self.__rec

    def __IndStat(self, func, method):
        resp_l = self.__resp_l
        ind_rs = layer_0.Target0()
        ind_rs.rr = func([i.rr for i in resp_l], method)
        ind_rs.v_t = func([i.v_t_i for i in resp_l], method)
        ind_rs.ve = func([i.ve for i in resp_l], method)
        ind_rs.wob = func([i.wob for i in resp_l], method)
        ind_rs.rsbi = func([i.rsbi for i in resp_l], method)
        ind_rs.mp_jl_d = func([i.mp_jl_d for i in resp_l], method)
        ind_rs.mp_jl_t = func([i.mp_jl_t for i in resp_l], method)
        ind_rs.mp_jm_d = func([i.mp_jm_d for i in resp_l], method)
        ind_rs.mp_jm_t = func([i.mp_jm_t for i in resp_l], method)
        return ind_rs

    def CountAggr(self, cate_l):
        for cate in cate_l:
            if cate == 'TD':
                self.TDAggr()
            elif cate == 'HRA':
                self.HRAAggr()
            elif cate == 'HRV':
                self.HRVAggr()
            elif cate == 'FD':
                self.FDAgger()
            else:
                print('No match category !')
                return

    def TDAggr(self):
        p_count = VarAnalysis().TimeSeries
        self.__rec.td.ave = self.__IndStat(p_count, 'AVE')
        self.__rec.td.med = self.__IndStat(p_count, 'MED')
        self.__rec.td.std = self.__IndStat(p_count, 'STD')
        self.__rec.td.cv = self.__IndStat(p_count, 'CV')
        self.__rec.td.qua = self.__IndStat(p_count, 'QUA')
        self.__rec.td.tqua = self.__IndStat(p_count, 'TQUA')

    # def FDAgger(self):
    #     p_count = VarAnalysis().FreqSeries
    #     self.__rec.fd.prsa = self.__IndStat()

    def HRAAggr(self):
        p_count = VarAnalysis().HRA
        self.__rec.hra.pi = self.__IndStat(p_count, 'PI')
        self.__rec.hra.gi = self.__IndStat(p_count, 'GI')
        self.__rec.hra.si = self.__IndStat(p_count, 'SI')

    def HRVAggr(self):
        p_count = VarAnalysis().HRV
        self.__rec.hrv.sd1 = self.__IndStat(p_count, 'SD1')
        self.__rec.hrv.sd2 = self.__IndStat(p_count, 'SD2')
