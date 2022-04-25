import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from scipy.stats import normaltest, ttest_ind, wilcoxon

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_p
from Classes.TypesInstant import RecordInfo
from Classes.Func.DiagramsGen import PlotMain
from Classes.ExtractSplice import ExtractSplice
from Classes.Func.CalculatePart import FreqPreMethod
from Classes.Func.KitTools import TimeShift, ConfigRead, SaveGen

# form_name = r'C:\Users\HY_Burger\Desktop\Project\extube_sump12.csv'
form_name = 'prepare.csv'
data_loc = Path(ConfigRead('WaveData', 'Extube'))
save_loc = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), 'PRSA')
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
    GraphProcess(p_list, 'AUC')


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


def GraphProcess(p_list: list, method: str) -> None:
    ture_arr = np.array([i['end'] for i in p_list])
    for ind in list(p_list[0]['prsa'].keys())[indicator_slice]:
        l_df = [pd.DataFrame(p['prsa'][ind]) for p in p_list]
        df_o = l_df[0].copy()
        df_o = df_o.reindex(index=df_o.index[::-1])
        for i in df_o.index:
            for j in df_o.columns:
                pred_arr = np.array([x.loc[i, j] for x in l_df])
                df_r = pd.DataFrame({'label': ture_arr, 'value': pred_arr})
                if method == 'AUC':
                    roc = roc_auc_score(ture_arr, pred_arr)
                    df_o.loc[i, j] = roc
                elif method == 'PV':
                    pred_arr_0 = df_r[df_r.label == 0].value
                    pred_arr_1 = df_r[df_r.label == 1].value
                    p, _, _ = PCount(pred_arr_0, pred_arr_1)
                    df_o.loc[i, j] = p
        p_ = PlotMain(save_loc)
        p_.HeatMapPlot(df_o, 'PRSA({0})'.format(ind))


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
    pid_obj.ridrec = rid_p.rec
    id_list = gp.get_group(pid).zdt_1.tolist()
    process_0 = ExtractSplice(pid_obj.ridrec)
    process_0.RecBatchesExtract(id_list, 1800)
    pid_obj.resp_l = process_0.RespSplicing(1800, vm_list)
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
    # re_rate = 4
    # L = 240
    # S_s = [2, 4, 6, 8, 10, 12, 14]
    # T_s = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    re_rate = 0.25
    L = 15
    S_s = [2, 4, 6]
    T_s = [1, 5, 10]

    result_d = {}
    p_ = PRSA(L, arr_v)
    p_.ReSample(re_rate, arr_t)
    for t in T_s:
        result_d[t] = {}
        p_.PRSAGet(t, 'AC')
        for s in S_s:
            p_.WaveletsAna(s)
            result_d[t][s] = p_.value
    return result_d


def PCount(arr_1, arr_2):
    _, p_1 = normaltest(arr_1)
    _, p_2 = normaltest(arr_2)
    alpha = 0.05
    if p_1 > alpha and p_2 > alpha:
        _, p = ttest_ind(arr_1, arr_2, equal_var=False)
        ave_1 = round(np.mean(arr_1), 3)
        std_1 = round(np.std(arr_1), 3)
        ave_2 = round(np.mean(arr_2), 3)
        std_2 = round(np.std(arr_2), 3)
        v_rs_1 = '{0} +- {1}'.format(ave_1, std_1)
        v_rs_2 = '{0} +- {1}'.format(ave_2, std_2)
        # p = round(p, 4) if p > 0.0001 else 'p < 0.0001'
    else:
        _, p = mannwhitneyu(arr_1, arr_2)
        med_1 = round(np.median(arr_1), 3)
        qua_1 = round(np.percentile(arr_1, 25), 3)
        tqua_1 = round(np.percentile(arr_1, 75), 3)
        med_2 = round(np.median(arr_2), 3)
        qua_2 = round(np.percentile(arr_2, 25), 3)
        tqua_2 = round(np.percentile(arr_2, 75), 3)
        v_rs_1 = '{0} ({1}, {2})'.format(med_1, qua_1, tqua_1)
        v_rs_2 = '{0} ({1}, {2})'.format(med_2, qua_2, tqua_2)
        # p = round(p, 4) if p > 0.0001 else 'p < 0.0001'
    p = round(p, 4)
    return p, v_rs_1, v_rs_2


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
        # p_.InterpValue(100)
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