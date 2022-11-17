import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from operator import add, mul
from WaveDataProcess import BinImport
from Classes.Domain import layer_0, layer_1, layer_2
from Classes.Func.KitTools import PathVerify, LocatSimiTerms
from Classes.Func.CalculatePart import IndCalculation, VarCount, TD, HRA, HRV, ENT, PRSA

ind_pasa_st = {
    'rr': {
        'L': 120,
        'F': 2.0,
        'T': 1,
        's': 6
    },
    'v_t': {
        'L': 120,
        'F': 2.0,
        'T': 5,
        's': 6
    },
    've': {
        'L': 120,
        'F': 2.0,
        'T': 45,
        's': 14
    },
    'mp_jb': {
        'L': 120,
        'F': 2.0,
        'T': 1,
        's': 6
    },
    'rsbi': {
        'L': 120,
        'F': 2.0,
        'T': 1,
        's': 14
    },
    'mp': {
        'L': 120,
        'F': 2.0,
        'T': 45,
        's': 14
    }
}


class Basic():
    def __init__(self) -> None:
        pass

    def __RecPathGen(self, parent_p: Path, rec: any) -> Path:

        if type(rec) == layer_1.RecWave:
            id_ = rec.zdt
            rec_p = PathVerify(parent_p) / (id_ + '.zdt')
        elif type(rec) == layer_1.RecPara:
            id_ = rec.zpx
            rec_p = PathVerify(parent_p) / (id_ + '.zpx')

        if rec_p.is_file():
            return rec_p
        else:
            t_ = rec.rec_t
            t_folder = str(t_.year) + str(t_.month).rjust(2, '0')
            rec_p = rec_p.parents[2] / t_folder / rec_p.parts[-2] / rec_p.name

            return rec_p


class RecordInfo(Basic):
    def __init__(self, id_: str, t_: datetime):
        super().__init__()
        self.__rec = layer_2.RidRec()
        self.__rec.zif = id_
        self.__rec.cls_t = t_  # give tmp t not close t

    @property
    def rec(self):
        return self.__rec

    def __PathCompose(self):
        id_ = self.__rec.zif
        t_ = self.__rec.cls_t
        t_folder = Path(str(t_.year) + str(t_.month).rjust(2, '0'))
        t_file = t_folder / id_ / (id_ + '.zif')
        return t_file

    def ParametersInit(self, parent_p: Path, opt: bool):
        rec = self.__rec
        rec.op_t = opt
        rec.zif = PathVerify(parent_p) / self.__PathCompose()
        p_rid = BinImport.RidData(self.__rec.zif)
        info = p_rid.RecordInfoGet()
        recs = p_rid.RecordListGet()
        try:
            rec.vm_n = info['m_n']
            rec.cls_t = max(recs['s_t'])
        except:
            print('zif file exist errors')
            rec = None


class RecordResp(Basic):
    def __init__(self, id_: str, t_: datetime):
        super().__init__()
        self.__rec = layer_1.RecWave()
        self.__rec.zdt = id_
        self.__rec.rec_t = t_

    @property
    def rec(self):
        return self.__rec

    def WaveformInit(self, parent_p: Path):
        rec = self.__rec
        rec.zdt = self._Basic__RecPathGen(parent_p, rec)
        p_wave = BinImport.WaveData(rec.zdt)
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
                resp_t = p_count.RespT(rec.sr)
                resp.wid = resp_t['wid']
                resp.t_i = resp_t['t_i']
                resp.t_e = resp_t['t_e']
                resp.i_e = resp_t['i_e']

                resp.pip, pip_val = p_count.PIP()
                resp.peep, peep_val = p_count.PEEP()
                resp.rr, rr_val = p_count.RR(rec.sr)

                v_t, v_t_val = p_count.V_T()
                resp.v_t_i = v_t['v_t_i']
                resp.v_t_e = v_t['v_t_e']

                resp.ve, ve_val = p_count.VE(resp.rr, resp.v_t_i)
                resp.rsbi = p_count.RSBI(resp.rr, resp.v_t_i)

                mp_jb, mp_jb_val = p_count.MP_Breath()
                resp.mp_jb_d = mp_jb['mp_jb_d']
                resp.mp_jb_t = mp_jb['mp_jb_t']
                resp.mp_jb_a = mp_jb['mp_jb_a']
                resp.mp_jb_b = mp_jb['mp_jb_b']

                mp_d, mp_d_val = p_count.MP_Area(resp.rr, resp.v_t_i,
                                                 resp.mp_jb_d)
                resp.mp_jm_d = mp_d['mp_jm']
                resp.mp_jl_d = mp_d['mp_jl']

                mp_t, mp_t_val = p_count.MP_Area(resp.rr, resp.v_t_i,
                                                 resp.mp_jb_t)
                resp.mp_jm_t = mp_t['mp_jm']
                resp.mp_jl_t = mp_t['mp_jl']

                val_list = [
                    rr_val, v_t_val, ve_val, mp_jb_val, mp_d_val, mp_t_val
                ]
                resp.val = True if not False in val_list else False

            rec.resps.append(resp)


class RecordPara(Basic):
    def __init__(self, id_: str, t_: datetime):
        super().__init__()
        self.__rec = layer_1.RecPara()
        self.__rec.zpx = id_
        self.__rec.rec_t = t_

    @property
    def rec(self):
        return self.__rec

    def __FieldVal(self, para, key_):
        try:
            value = para[key_]
        except:
            value = None
        return value

    def ParametersInit(self, parent_p: Path, machine: str):
        rec = self.__rec
        rec.zpx = self._Basic__RecPathGen(parent_p, rec)
        p_para = BinImport.ParaData(rec.zpx)
        para = p_para.ParaInfoGet()
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
                rec.st_mode = p_para.VMInter(machine, slice(0, len(rec.u_ind)))
            except:
                rec.st_mode = []

    def ParaSelectBT(self, SR, time_tag, para_l):
        rec = self.__rec
        t_ind = list(map(mul, rec.u_ind, [1 / SR] * len(rec.u_ind)))
        s_ind = LocatSimiTerms(t_ind, time_tag).values()
        p_s_l = [para_l[i] for i in s_ind if i != None]
        return p_s_l


class ResultStatistical(Basic):
    def __init__(self, resp_list: list) -> None:
        super().__init__()
        self.__resp_l = resp_list
        self.__rec = layer_2.Result()
        self.__rec.td = layer_1.DomainTS()
        self.__rec.fd = layer_1.DomainFS()
        self.__rec.hra = layer_1.DomainHRA()
        self.__rec.hrv = layer_1.DomainHRV()
        self.__rec.ent = layer_1.DomainEntropy()
        self.__rec.prsa = layer_1.DomainPRSA()

        self.__prsa_dist = {}

    @property
    def rec(self):
        return self.__rec

    @property
    def prsa_dist(self):
        return self.__prsa_dist

    def __IndStat(self, cls_: any, method_s: list, para_: dict = {}) -> dict:
        ind_sel = ['rr', 'v_t', 've', 'mp_jb', 'rsbi', 'mp']
        para_ = dict.fromkeys(ind_sel, {}) if not para_ else para_
        ind_l = lambda n: [getattr(i, n) for i in self.__resp_l]

        var_counter = {
            't_tot': cls_(ind_l('wid'), **para_['rr']),
            't_i': cls_(ind_l('t_i'), **para_['rr']),
            't_e': cls_(ind_l('t_e'), **para_['rr']),
            'i_e': cls_(ind_l('i_e'), **para_['rr']),
            'pip': cls_(ind_l('pip'), **para_['mp_jb']),
            'peep': cls_(ind_l('peep'), **para_['mp_jb']),
            'rr': cls_(ind_l('rr'), **para_['rr']),
            'v_t': cls_(ind_l('v_t_i'), **para_['v_t']),
            've': cls_(ind_l('ve'), **para_['ve']),
            'rsbi': cls_(ind_l('rsbi'), **para_['rsbi']),
            'mp_jb_d': cls_(ind_l('mp_jb_d'), **para_['mp_jb']),
            'mp_jb_t': cls_(ind_l('mp_jb_t'), **para_['mp_jb']),
            'mp_jl_d': cls_(ind_l('mp_jl_d'), **para_['mp']),
            'mp_jl_t': cls_(ind_l('mp_jl_t'), **para_['mp']),
            'mp_jm_d': cls_(ind_l('mp_jm_d'), **para_['mp']),
            'mp_jm_t': cls_(ind_l('mp_jm_t'), **para_['mp']),
        }

        var_s = []

        for i in range(len(method_s)):
            var_ = layer_0.Target0()
            for k, v in var_counter.items():
                setattr(var_, k, getattr(v, method_s[i]))
            var_s.append(var_)

        var_d = dict(zip(method_s, var_s))

        return var_d

    def CountAggr(self, cate_l: list, **kwargs):
        for cate in cate_l:
            if cate == 'td':
                self.TDAggr()
            elif cate == 'hra':
                self.HRAAggr()
            elif cate == 'hrv':
                self.HRVAggr()
            elif cate == 'ent':
                self.EntAggr()
            elif cate == 'prsa':
                self.PRSAAggr(**kwargs)
            else:
                print('No match category !')
                return

    def TDAggr(self) -> None:
        sub_m = ['ave', 'med', 'std', 'cv', 'qua', 'tqua']
        rs = self.__IndStat(TD, sub_m)
        self.__rec.td.ave = rs['ave']
        self.__rec.td.med = rs['med']
        self.__rec.td.std = rs['std']
        self.__rec.td.cv = rs['cv']
        self.__rec.td.qua = rs['qua']
        self.__rec.td.tqua = rs['tqua']

    def HRAAggr(self) -> None:
        sub_m = ['pi', 'gi', 'si']
        rs = self.__IndStat(HRA, sub_m)
        self.__rec.hra.pi = rs['pi']
        self.__rec.hra.gi = rs['gi']
        self.__rec.hra.si = rs['si']

    def HRVAggr(self) -> None:
        sub_m = ['sd1', 'sd2']
        rs = self.__IndStat(HRV, sub_m)
        self.__rec.hrv.sd1 = rs['sd1']
        self.__rec.hrv.sd2 = rs['sd2']

    def EntAggr(self) -> None:
        sub_m = ['apen', 'sampen', 'fuzzen']
        rs = self.__IndStat(ENT, sub_m)
        self.__rec.ent.app = rs['apen']
        self.__rec.ent.samp = rs['sampen']
        self.__rec.ent.fuzz = rs['fuzzen']

    def PRSAAggr(self, para_st: dict = {}) -> None:
        resp_wid = [round(i.wid) for i in self.__resp_l]

        ind_para = dict(
            zip(list(ind_pasa_st.keys()), [para_st] *
                len(ind_pasa_st.keys()))) if para_st else ind_pasa_st

        for k in ind_para.keys():
            ind_para[k]['ind_s'] = resp_wid

        sub_m = ['ac', 'mean_ac', 'dc', 'mean_dc']
        rs = self.__IndStat(PRSA, sub_m, ind_para)

        self.__rec.prsa.ac = rs['ac']
        self.__rec.prsa.dc = rs['dc']
        self.__prsa_dist['ac'] = rs['mean_ac']
        self.__prsa_dist['dc'] = rs['mean_dc']