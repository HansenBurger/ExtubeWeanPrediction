from unittest import result
import pandas as pd
import instantiation as instan
from datetime import datetime
from pathlib import Path
from Classes.Func import kit
from operator import mul, sub, add


def main():
    df = pd.read_csv('1.csv')
    data_loc = Path(kit.ConfigRead('WaveData', 'Extube'))
    kit.TimeShift(df, ['endo_t', 'END_t', 'Resp_t'])
    gp = df.groupby('PID')
    results_d = kit.FromkeysReid(['pid', 'end', 'var_rs'])

    for pid in df.PID.unique():
        t_s = datetime.now()
        df_tmp = PatientGp(gp, pid)
        pid_rec = RecDataGen(data_loc, df_tmp)
        resp_l = SelectByTime(1800, df_tmp, pid_rec)

        if not resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            var_rs = VarRsGen(resp_l, ['TD', 'HRA', 'HRV'])

            results_d['pid'].append(pid)
            results_d['end'].append(0 if '成功' in
                                    df_tmp.endo_end.unique()[0] else 1)
            results_d['var_rs'].append(var_rs)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))

    a = 1


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


def SelectByTime(t_set, df_in, sample):
    resp_select = []

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
    vm_sl = vm_l[0:kit.LocatSimiTerms(ut_s, [-t_set])[-t_set]]
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