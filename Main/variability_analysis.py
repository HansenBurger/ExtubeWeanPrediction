import sys
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_p
from Classes.TypesInstant import RecordInfo
from Classes.ExtractSplice import ExtractSplice
from Classes.VarResultsGen import VarResultsGen
from Classes.Func.KitTools import ConfigRead, TimeShift, SaveGen, measure
from Classes.ORM.basic import db
from Classes.ORM.expr import PatientInfo
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12, WeanPSV, WeanSumP12

mode_ = 'Extube_PSV_Nad'
mode_info = {
    'Extube': {
        'PSV':
        ExtubePSV,
        'SumP12':
        ExtubeSumP12,
        'multirid': [
            2417319, 5597021, 5616583, 5694206, 6447979, 6584930, 6750903,
            6828867
        ]
    },
    'Wean': {
        'PSV':
        WeanPSV,
        'SumP12':
        WeanSumP12,
        'multirid': [
            2417319, 3351693, 4151492, 4281275, 4455851, 5597021, 5628925,
            5694206, 6396656, 6447142, 6447979, 6750903, 6828867, 6983384
        ]
    }
}

data_loc = Path(ConfigRead('WaveData', mode_.split('_')[0]))
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), mode_)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), mode_)

col_range_set = {
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

vm_list = ['SPONT', 'CPAP', 'APNEA VENTILATION']
p_trend_l = [
    'bed_sbp', 'bed_dbp', 'bed_mbp', 'bed_spo2', 'bed_rr', 'bed_pr', 'bed_cvpm'
]
method_list = ['TD', 'HRA', 'HRV', 'ENT', 'PRSA']


@measure
def main():
    df_tot = TableQuery(len(mode_.split('_')) > 2)
    gp_tot = df_tot.groupby('pid')
    pid_list = df_tot.pid.unique()
    pid_o_l = PIDTest(gp_tot, pid_list)
    PInfoCollect(pid_o_l)


def PIDTest(gp, pids):
    pid_dr_s = []

    for pid in pids:
        t_s = datetime.now()
        pid_obj = DataGen(gp, pid)
        pid_dr_s.append({'validy': pid_obj.validy, 'end': pid_obj.end_i})
        if not pid_obj.resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            process_1 = VarResultsGen(pid_obj)
            process_1.VarRsGen(method_list)
            process_1.TensorStorage(s_f_fold)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))

    return pid_dr_s


def DataGen(gp, pid):
    df = gp.get_group(pid)
    df = df.reset_index(drop=True)
    rid = df.rid.unique()[-1]

    pid_o = layer_p.Patient()
    pid_o.pid = pid
    pid_o.icu = df.icu[0]
    pid_o.end_t = df.e_t[0]
    pid_o.end_i = df.e_s[0]

    rid_p = RecordInfo(rid, pid_o.end_t)
    rid_p.ParametersInit(data_loc, df.opt[0])
    pid_o.rid_s = rid_p.rec

    rec_id_s = df.zdt.tolist()
    rec_t_s = df.rec_t.tolist()

    splice_p = ExtractSplice(pid_o.rid_s)
    splice_p.RecBatchesExtract(rec_id_s, rec_t_s, 1800)
    pid_o.resp_l, pid_o.validy = splice_p.RespSplicing(vm_list, 1800)

    return pid_o


def PInfoCollect(pid_dr_l: list):
    save_path = s_g_fold / 'process_info.txt'
    save_param = {
        'val_succ_n': 0,
        'val_fail_n': 0,
        'data_inval': 0,
        'mode_inval': 0,
        'mode_inval_succ': 0,
        'mode_inval_fail': 0
    }
    repr_gen = lambda dict_: ('\n').join(k + ':\t' + str(v)
                                         for k, v in dict_.items())
    for p_dr in pid_dr_l:
        p_dr_val = reduce(lambda x, y: x & y, list(p_dr['validy'].values()))

        end_con_0 = not p_dr['end']
        end_con_1 = p_dr['end']
        data_con = sum(p_dr['validy'].values()) == 2
        mode_con = sum(p_dr['validy'].values()) == 1

        if p_dr_val:
            save_param['val_succ_n'] += 1 if end_con_0 else 0
            save_param['val_fail_n'] += 1 if end_con_1 else 0
        else:
            save_param['data_inval'] += 1 if data_con else 0
            save_param['mode_inval'] += 1 if mode_con else 0
            save_param['mode_inval_succ'] += 1 if mode_con and end_con_0 else 0
            save_param['mode_inval_fail'] += 1 if mode_con and end_con_1 else 0

    param_repr = repr_gen(save_param)

    with open(save_path, 'w') as f:
        f.write(mode_ + '_ProcessInfo:\n')
        f.write(param_repr)


def TableQuery(aged_ill: bool = False) -> pd.DataFrame:
    mode_n_l = mode_.split('_')
    src_ = mode_info[mode_n_l[0]][mode_n_l[1]]
    if aged_ill:
        cond = src_.pid.in_(NonAgedIllQuery())
    else:
        cond = src_.pid > 0

    cond_ = ~src_.pid.in_(mode_info[mode_n_l[0]]['multirid'])

    que = src_.select().where(cond & cond_)
    df = pd.DataFrame(list(que.dicts()))
    df = df.drop('index', axis=1)
    df.e_s = np.where(df.e_s.str.contains('成功'), 0, 1)
    return df


def NonAgedIllQuery():
    mode_n_l = mode_.split('_')
    src_0 = PatientInfo
    src_1 = mode_info[mode_n_l[0]][mode_n_l[1]]
    join_info = {'dest': src_0, 'on': src_0.pid == src_1.pid, 'attr': 'pinfo'}
    cond_age = src_0.age <= 75
    col_rmk = [src_0.rmk, src_0.rmk_i, src_0.rmk_i_, src_0.rmk_o, src_0.rmk_o_]
    cond_d_f = lambda x: (x.is_null()) | (~x.contains('脑') & ~x.contains('神经'))
    cond_d_rmk = reduce(lambda x, y: x & y, [cond_d_f(col) for col in col_rmk])
    que_l = src_1.select(src_1.pid).join(**join_info).where(cond_age
                                                            & cond_d_rmk)
    pid_l = [que.pid for que in que_l.group_by(src_1.pid)]
    return pid_l


if __name__ == '__main__':
    main()