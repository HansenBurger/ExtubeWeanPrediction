import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_3
from Classes.TypesInstant import RecordInfo
from Classes.ExtractSplice import ExtractSplice
from Classes.VarResultsGen import VarResultsGen
from Classes.Func.KitTools import ConfigRead, TimeShift, SaveGen

mode_name = 'extube_sump12'
file_name = r'C:\Users\HY_Burger\Desktop\Project\extube_sump12.csv'
data_loc = Path(ConfigRead('WaveData', 'Extube'))
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), mode_name)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), mode_name)

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
method_list = ['TD', 'HRA', 'HRV']


def main():
    df = pd.read_csv(file_name)
    df.endo_end = np.where(df.endo_end.str.contains('成功'), 0, 1)
    TimeShift(df, ['endo_t', 'END_t', 'Resp_t'])
    gp = df.groupby('PID')

    for pid in df.PID.unique():
        t_s = datetime.now()
        id_list = gp.get_group(pid).zdt_1.tolist()
        pid_obj = PatientGen(gp, pid)
        process_0 = ExtractSplice(pid_obj.ridrec)
        process_0.RecBatchesExtract(id_list, 1800)
        pid_obj.resp_l = process_0.RespSplicing(1800, vm_list)

        if not pid_obj.resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            process_1 = VarResultsGen(pid_obj)
            process_1.VarRsGen(method_list)
            process_1.TensorStorage(s_f_fold)
            # process_1.TrendsPlot(s_g_fold)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))


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


if __name__ == '__main__':
    main()