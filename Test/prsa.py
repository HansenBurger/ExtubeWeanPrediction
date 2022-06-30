import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_p
from Classes.TypesInstant import RecordInfo
from Classes.Func.DiagramsGen import PlotMain
from Classes.ExtractSplice import ExtractSplice
from Classes.Func.CalculatePart import FreqPreMethod, PerfomAssess
from Classes.Func.KitTools import TimeShift, ConfigRead, SaveGen

mode_name = 'PRSA_AC_s'
form_name = 'prepare.csv'
# form_name = r'C:\Users\HY_Burger\Desktop\Project\extube_sump12.csv'
data_loc = Path(ConfigRead('WaveData', 'Extube'))
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), mode_name)
s_g_loc = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), mode_name)
vm_list = ['SPONT', 'CPAP', 'APNEA VENTILATION']
indicator_slice = slice(1, 7)


def main():
    p_list = []
    pid_l, gp_ = Preprocess(form_name)
    for pid in pid_l:
        prsa_d = CountPorcess(gp_, pid)
        if not prsa_d:
            pass
        else:
            p_list.append(prsa_d)
    GraphProcess(p_list)


def Preprocess(form_n):
    df_main = pd.read_csv(form_n)
    df_main.endo_end = np.where(df_main.endo_end.str.contains('成功'), 0, 1)
    TimeShift(df_main, ['endo_t', 'END_t', 'Resp_t'])
    gp_main = df_main.groupby('PID')
    pid_list = df_main.PID.unique()
    return pid_list, gp_main


def CountPorcess(gp, pid):
    t_s = datetime.now()
    pr_d = {}
    end_i, resp_l = InfoCollect(gp, pid)
    if not resp_l:
        print('{0}\' has no valid data'.format(pid))
        return None
    else:
        resp_d = ValueGen(resp_l)
        for cate in list(resp_d.keys())[indicator_slice]:
            result_d = PRSARangeTest(resp_d['t_ind'], resp_d[cate])
            resp_d[cate] = result_d
        pr_d['pid'] = pid
        pr_d['end'] = end_i
        pr_d['prsa'] = resp_d
        t_e = datetime.now()
        print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))
    return pr_d


def GraphProcess(p_list: list) -> None:
    ture_arr = np.array([i['end'] for i in p_list])
    for ind in list(p_list[0]['prsa'].keys())[indicator_slice]:
        l_df = [pd.DataFrame(p['prsa'][ind]) for p in p_list]

        df_o = l_df[0].copy()
        df_p = df_o.reindex(index=df_o.index[::-1])
        df_p_pos = df_p.copy()
        df_p_neg = df_p.copy()
        df_auc = df_p.copy()

        for i in df_o.index:
            for j in df_o.columns:
                pred_arr = np.array([x.loc[i, j] for x in l_df])
                p_assess = PerfomAssess(ture_arr, pred_arr)
                roc, _, _ = p_assess.AucAssess()
                df_auc.loc[i, j] = roc
                p, rs_pos, rs_neg = p_assess.PValueAssess()
                df_p.loc[i, j] = p
                df_p_pos.loc[i, j] = rs_pos
                df_p_neg.loc[i, j] = rs_neg

        pd.DataFrame.to_csv(df_p_pos,
                            s_f_fold / 'p_pos_dist_{0}.csv'.format(ind),
                            index=False)
        pd.DataFrame.to_csv(df_p_neg,
                            s_f_fold / 'p_neg_dist_{0}.csv'.format(ind),
                            index=False)
        p_ = PlotMain(s_g_loc)
        p_.HeatMapPlot(df_auc, 'PRSA_AUC({0})'.format(ind))
        p_.HeatMapPlot(df_p, 'PRSA_P({0})'.format(ind))


def InfoCollect(gp, pid):
    df = gp.get_group(pid)
    df = df.reset_index(drop=True)
    rid = df.Record_id.unique()[0]
    pid_obj = layer_p.Patient()
    pid_obj.pid = pid
    pid_obj.end_t = df.endo_t[0]
    pid_obj.end_i = df.endo_end.unique()[0]
    rid_p = RecordInfo(data_loc, pid_obj.end_t, rid)
    rid_p.ParametersInit()
    pid_obj.rid_s = rid_p.rec
    id_list = gp.get_group(pid).zdt_1.tolist()
    process_0 = ExtractSplice(pid_obj.rid_s)
    process_0.RecBatchesExtract(id_list, 1800)
    pid_obj.resp_l, _ = process_0.RespSplicing(vm_list, 1800)
    return pid_obj.end_i, pid_obj.resp_l


def ValueGen(resp_l):
    rr = np.array([i.rr for i in resp_l])
    v_t = np.array([i.v_t_i for i in resp_l])
    ve = np.array([i.ve for i in resp_l])
    wob = np.array([i.wob for i in resp_l])
    rsbi = np.array([i.rsbi for i in resp_l])
    mp_jl_t = np.array([i.mp_jl_t for i in resp_l])
    mp_jm_t = np.array([i.mp_jm_t for i in resp_l])
    wid_l = [round(i.wid) for i in resp_l]
    t_ind = np.array([sum(wid_l[0:i]) for i in range(len(wid_l))])
    dict_ = {
        't_ind': t_ind,
        'rr': rr,
        'v_t': v_t,
        've': ve,
        'wob': wob,
        'rsbi': rsbi,
        'mp_jm_t': mp_jm_t,
        'mp_jl_t': mp_jl_t
    }
    return dict_


def PRSARangeTest(arr_t, arr_v):
    re_rate = 2
    L = 120
    S_s = [2, 4, 6, 8, 10, 12, 14]
    T_s = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # re_rate = 0.25
    # L = 15
    # S_s = [2, 4, 6]
    # T_s = [1, 5, 10]

    result_d = {}
    p_ = PRSA(L, arr_v)
    p_.ReSample(re_rate, arr_t)
    for t in T_s:
        result_d[t] = {}
        p_.PRSAGet(t, 'DC')
        for s in S_s:
            p_.WaveletsAna(s)
            result_d[t][s] = p_.value
    return result_d


class PRSA():
    def __init__(self, L, arr):
        self.__L = L
        self.__arr_i = arr
        self.__prsa_d = None
        self.__value = -1

    @property
    def value(self):
        return self.__value

    def __WtJudge(self, val):
        if val >= -1 and val < 0:
            para = -1 / 2
        elif val >= 0 and val < 1:
            para = 1 / 2
        else:
            para = 0
        return para

    def ReSample(self, re_rate, arr_t):
        p_ = FreqPreMethod(arr_t, self.__arr_i)
        p_.InitTimeSeries()
        p_.Resampling(re_rate)
        self.__arr_i = p_.df.value

    def PRSAGet(self, T, method):
        L = self.__L
        anchor_s = []
        if method == 'AC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) > np.mean(x[
                y - T:y]) else False
        elif method == 'DC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) < np.mean(x[
                y - T:y]) else False
        else:
            print('No match method')
            return

        for i in range(L, len(self.__arr_i) - L):
            if not anchor_set(self.__arr_i, i):
                pass
            else:
                clip = slice(i - L, i + L + 1)
                anchor_s.append(self.__arr_i[clip].tolist())
        arr_prsa = np.array([np.mean(i) for i in np.array(anchor_s).T])
        arr_axis = np.linspace(-L, L + 1, 2 * L + 1, endpoint=False)
        df = pd.DataFrame({'axis': arr_axis, 'prsa': arr_prsa})
        df = df.set_index('axis', drop=True)
        self.__prsa_d = df

    def WaveletsAna(self, s):
        df = self.__prsa_d.copy()
        axis_s = np.linspace(-s, s, 2 * s, endpoint=False)
        prsa_s = df.loc[axis_s].prsa
        para_s = np.array([self.__WtJudge(i / s) for i in axis_s])
        value = np.sum(prsa_s * para_s / s)
        self.__value = value


if __name__ == '__main__':
    main()