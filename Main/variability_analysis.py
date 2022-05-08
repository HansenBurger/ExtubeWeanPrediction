import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_p
from Classes.TypesInstant import RecordInfo
from Classes.ExtractSplice import ExtractSplice
from Classes.VarResultsGen import VarResultsGen
from Classes.Func.KitTools import ConfigRead, TimeShift, SaveGen
from Classes.ORM.basic import db
from Classes.ORM.expr import PatientInfo
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12, WeanPSV, WeanSumP12

mode_ = 'Extube_PSV'
mode_info = {
    'Extube': {
        'PSV': ExtubePSV,
        'SumP12': ExtubeSumP12
    },
    'Wean': {
        'PSV': WeanPSV,
        'SumP12': WeanSumP12
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


def main():
    df = TableQuery()
    gp = df.groupby('pid')
    pid_list = df.pid.unique()
    PIDTest(gp, pid_list)


def PIDTest(gp, pids):
    for pid in pids:
        t_s = datetime.now()
        id_list = gp.get_group(pid).zdt.tolist()
        pid_obj = PatientGen(gp, pid)
        process_0 = ExtractSplice(pid_obj.ridrec)
        process_0.RecBatchesExtract(id_list, 1800)
        pid_obj.resp_l = process_0.RespSplicing(vm_list, 1800)

        if not pid_obj.resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            process_1 = VarResultsGen(pid_obj)
            process_1.VarRsGen(method_list)
            process_1.TensorStorage(s_f_fold)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))


def PatientGen(gp, pid):
    df = gp.get_group(pid)
    df = df.reset_index(drop=True)
    rid = df.rid.unique()[0]

    pid_o = layer_p.Patient()
    pid_o.pid = pid
    pid_o.end_t = df.e_t[0]
    pid_o.end_i = df.e_s[0]

    rid_p = RecordInfo(rid, pid_o.end_t)
    rid_p.ParametersInit(data_loc, df.opt[0])
    pid_o.ridrec = rid_p.rec

    return pid_o


def TableQuery():
    mode_n_l = mode_.split('_')
    src_ = mode_info[mode_n_l[0]][mode_n_l[1]]
    que = src_.select()
    df = pd.DataFrame(list(que.dicts()))
    df = df.drop('index', axis=1)
    df.e_t = np.where(df.e_t.str.contains('成功'), 0, 1)
    return df


if __name__ == '__main__':
    main()