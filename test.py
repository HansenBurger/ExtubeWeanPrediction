import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from operator import mul, sub, add
import Classes.TypesInstant as instan
from Classes.Func import DiagramsGen, KitTools

mode_name = 'extube_sump12'


def main():
    df = pd.read_csv('extube_sump12.csv')[57:60]
    df.endo_end = np.where(df.endo_end.str.contains('成功'), 0, 1)
    KitTools.TimeShift(df, ['endo_t', 'END_t', 'Resp_t'])
    gp = df.groupby('PID')

    data_loc = Path(KitTools.ConfigRead('WaveData', 'Extube'))
    s_f_fold = KitTools.SaveGen(
        Path(KitTools.ConfigRead('ResultSave', 'Form')), mode_name)
    s_g_fold = KitTools.SaveGen(
        Path(KitTools.ConfigRead('ResultSave', 'Graph')), mode_name)

    result_l = []
    for pid in df.PID.unique():
        t_s = datetime.now()
        df_tmp = PatientGp(gp, pid)
        pid_rec = RecDataGen(data_loc, df_tmp)
        resp_l = SelectByTime(1800, df_tmp, pid_rec)

        if not resp_l:
            print('{0}\' has no valid data'.format(pid))
            result_l.append(None)
            continue
        else:
            var_rs = VarRsGen(resp_l, ['TD', 'HRA', 'HRV'])
            save_n = '{0}_{1}_{2}'.format(df_tmp.PID[0], df_tmp.endo_end[0],
                                          df_tmp.Record_id[0])

            #IndicatorsTrends(resp_l, s_g_fold, save_n)
            TensorStorage(df_tmp, var_rs, s_f_fold, save_n)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))


def PatientGp(gp_in, pid):
    df_ = gp_in.get_group(pid)
    df_ = df_.reset_index(drop=True)
    df_ = df_.loc[::-1].reset_index(drop=True)
    return df_


def RecDataGen(folder, df_in):
    end_t = df_in.endo_t[0]
    rid_s = df_in.Record_id.unique()[0]
    rid_p = instan.RecordInfo(folder, end_t, rid_s)
    rid_p.ParametersInit()
    return rid_p.rec


def WaveGen(folder, id_):
    wav_p = instan.RecordResp(folder, id_)
    wav_p.WaveformInit()
    wav_p.IndicatorCalculate()
    return wav_p.rec


def ParaGen(folder, id_, vm):
    par_p = instan.RecordPara(folder, id_)
    par_p.ParametersInit(vm)
    return par_p.rec


def VarRsGen(resp_val_l, method_l):
    res_p = instan.ResultStatistical(resp_val_l)
    res_p.CountAggr(method_l)
    return res_p.rec


def IndicatorsTrends(resp_l, s_g_loc, save_n):
    save_name = save_n
    wid_l = [i.wid for i in resp_l]
    stl_l = [sum(wid_l[0:i]) for i in range(1, len(wid_l) + 1)]
    df = pd.DataFrame([KitTools.GetObjectDict(i) for i in resp_l])
    df['ind'] = stl_l
    col_sel = {
        'rr': [],
        'v_t_i': [100, 800],
        've': [],
        'rsbi': [0, 220],
        'wob': [],
        'mp_jm_d': [],
        'mp_jm_t': [],
        'mp_jl_d': [],
        'mp_jl_t': [0.7, 1.2]
    }
    DiagramsGen.PlotMain(s_g_loc).MultiLineplot('ind', col_sel, df, save_name)


def TensorStorage(var_rs, s_f_loc, save_n):

    var_sl = [var_rs.td, var_rs.hra, var_rs.hrv]
    var_sl = [KitTools.GetObjectDict(i) for i in var_sl]
    var_sd = {}
    for i in var_sl:
        var_sd.update(i)
    var_save = []
    for k, v in var_sd.items():
        dict_ = {'method': k}
        dict_.update(KitTools.GetObjectDict(v))
        var_save.append(dict_)
    df_out = pd.DataFrame(var_save).set_index(['method'])
    pd.DataFrame.to_csv(df_out, s_f_loc / (save_n + '.csv'))


def SelectByTime(t_set, df_in, sample):
    resp_select = []

    # data process

    v_still_t = 0
    wave_data = []
    para_data = []
    for i in range(df_in.shape[0]):
        if v_still_t > t_set:
            break
        w_d = WaveGen(sample.zif.parent, df_in.zdt_1[i])
        p_d = ParaGen(sample.zif.parent, df_in.zpx_1[i], sample.vm_n)
        v_still_t += sum([i.wid for i in w_d.resps if i.val])
        wave_data.append(w_d)
        para_data.append(p_d)

    if v_still_t < t_set:
        return None

    # data splicing

    ut_s = []
    vm_l = []
    for i in para_data:
        i.st_mode.reverse()
        vm_l.extend(i.st_mode)
        i.u_ind.reverse()
        ut_l = list(map(mul, i.u_ind, [1 / wave_data[0].sr] * len(i.u_ind)))
        ut_l = list(map(sub, ut_l, [ut_l[0]] * len(ut_l)))
        ut_l = list(map(add, ut_l, [ut_s[-1]] * len(ut_l))) if ut_s else ut_l
        ut_s.extend(ut_l)
    vm_sl = vm_l[0:KitTools.LocatSimiTerms(ut_s, [-t_set])[-t_set]]
    val_sl = [
        True
        if 'SPONT' in i or 'CPAP' in i or 'APNEA VENTILATION' in i else False
        for i in vm_sl
    ]

    if False in val_sl:
        return None

    v_still_t = 0
    for i in wave_data:
        i.resps.reverse()  # extube as start
        for j in i.resps:
            if v_still_t > t_set:
                break
            elif not j.val:
                continue
            else:
                v_still_t += j.wid
                resp_select.append(j)
    resp_select.reverse()
    return resp_select


if __name__ == '__main__':
    main()