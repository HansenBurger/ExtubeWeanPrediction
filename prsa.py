import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from Classes.Domain import layer_3
from Classes.TypesInstant import RecordInfo
from Classes.ExtractSplice import ExtractSplice
from Classes.Func.KitTools import TimeShift, ConfigRead
from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.CalculatePart import FreqPreMethod

file_name = r'C:\Users\HY_Burger\Desktop\Project\extube_sump12.csv'
data_loc = Path(ConfigRead('WaveData', 'Extube'))
vm_list = ['SPONT', 'CPAP', 'APNEA VENTILATION']
df_main = pd.read_csv(file_name)
df_main.endo_end = np.where(df_main.endo_end.str.contains('成功'), 0, 1)
TimeShift(df_main, ['endo_t', 'END_t', 'Resp_t'])
gp_main = df_main.groupby('PID')

re_rate = 4
L = 120


def main():
    p_list = []
    for pid in df_main.PID.unique()[0:10]:
        t_s = datetime.now()
        id_list = gp_main.get_group(pid).zdt_1.tolist()
        pid_obj = PatientGen(gp_main, pid)
        process_0 = ExtractSplice(pid_obj.ridrec)
        process_0.RecBatchesExtract(id_list, 1800)
        pid_obj.resp_l = process_0.RespSplicing(1800, vm_list)
        resp_l = pid_obj.resp_l
        if not resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            try:
                resp_d = ReSample(resp_l, re_rate)
            except:
                continue
            for key in resp_d.keys():
                if key == 't_ind':
                    pass
                else:
                    resp_d[key] = PRSACount(resp_d[key], L)
            del resp_d['t_ind']
            resp_d['pid'] = pid
            p_list.append(resp_d)
        t_e = datetime.now()
        print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))
    df = pd.DataFrame(p_list)
    pd.DataFrame.to_csv(df, 'xxxx.csv', index=False)


def PatientGen(gp, pid):
    df = gp.get_group(pid)
    df = df.reset_index(drop=True)
    rid = df.Record_id.unique()[0]
    pid_obj = layer_3.Patient()
    pid_obj.pid = pid
    pid_obj.end_t = df.endo_t[0]
    pid_obj.end_i = df.endo_end.unique()[0]
    rid_p = RecordInfo(data_loc, pid_obj.end_t, rid)
    rid_p.ParametersInit()
    pid_obj.ridrec = rid_p.rec
    return pid_obj


def RecordGet(gp, pid):
    id_list = gp.get_group(pid).zdt_1.tolist()
    pid_obj = PatientGen(gp, pid)
    process_0 = ExtractSplice(pid_obj.ridrec)
    process_0.RecBatchesExtract(id_list, 1800)
    pid_obj.resp_l = process_0.RespSplicing(1800, vm_list)
    return pid_obj.resp_l


def ReSample(resp_l, re_rate):
    rr = np.array([i.rr for i in resp_l])
    v_t = np.array([i.v_t_i for i in resp_l])
    ve = np.array([i.ve for i in resp_l])
    wob = np.array([i.wob for i in resp_l])
    rsbi = np.array([i.rsbi for i in resp_l])
    mp_jl_d = np.array([i.mp_jl_d for i in resp_l])
    wid_l = [round(i.wid) for i in resp_l]
    t_ind = np.array([sum(wid_l[0:i]) for i in range(len(wid_l))])
    dict_ = {
        't_ind': t_ind,
        'rr': rr,
        'v_t': v_t,
        've': ve,
        'wob': wob,
        'rsbi': rsbi,
        'mp_jl_d': mp_jl_d
    }

    for key in dict_.keys():
        if key == 't_ind':
            pass
        else:
            p_ = FreqPreMethod(dict_['t_ind'], dict_[key])
            p_.InterpValue(100)
            p_.Resampling(re_rate)
            dict_[key] = np.array(p_.df.value)
    return dict_


def PRSACount(v_a, L):
    v_s_l = []
    ac_anchor = lambda x, y: True if x[y] > x[y - 1] else False
    dc_anchor = lambda x, y: True if x[y] < x[y - 1] else False
    for i in range(L, len(v_a) - L):
        if not ac_anchor(v_a, i):
            pass
        else:
            clip = slice(i - L, i + L + 1)
            v_s = v_a[clip].tolist()
            v_s_l.append(v_s)
    v_prsa = np.array([np.mean(i) for i in np.array(v_s_l).T])
    x_axis = np.linspace(-L, L + 1, 2 * L + 1, endpoint=False)
    # df_prsa = pd.DataFrame({'time': x_axis, 'value': v_prsa})
    # ac_l = np.array([df_prsa[df_prsa.time == i].value
    #                  for i in [-2, -1, 0, 1]]).T[0]
    # ac_conv = (ac_l[2] + ac_l[3] - ac_l[1] - ac_l[0]) / 4
    df = pd.DataFrame({'axis': x_axis, 'value': v_prsa})
    df = df.set_index('axis', drop=True)
    conv_s = lambda x: df.loc[x].value
    conv = (conv_s(0) + conv_s(1) - conv_s(-1) - conv_s(-2)) / 4
    return round(conv, 4)


if __name__ == '__main__':
    main()