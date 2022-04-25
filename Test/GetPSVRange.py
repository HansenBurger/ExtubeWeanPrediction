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
from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.KitTools import ConfigRead, TimeShift, SaveGen

mode_name = 'wean_vm'
file_name = r'C:\Users\HY_Burger\Desktop\Project\Recordinfo_filted_with_.csv'
# file_name = 'prepare.csv'  #TODO change the file loc
data_loc = Path(ConfigRead('WaveData', 'Wean'))
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), mode_name)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), mode_name)

p_trend_l = ['st_mode']


def main():
    df = pd.read_csv(file_name)
    df.endo_end = np.where(df.endo_end.str.contains('成功'), 0, 1)
    TimeShift(df, ['endo_t', 'END_t', 'Resp_t'])
    gp = df.groupby('PID')
    a = df.PID.unique()
    PIDTest(gp, a)


def PIDTest(gp, pids):
    result_l = []
    for pid in pids:
        t_s = datetime.now()
        id_list = gp.get_group(pid).zdt_1.tolist()
        pid_obj = PatientGen(gp, pid)
        process_0 = ExtractSplice(pid_obj.ridrec)
        process_0.RecBatchesExtract(id_list)
        try:
            pid_obj.para_d = process_0.ParaSplicing(p_trend_l)
        except:
            print('{0}\' has no valid data'.format(pid))
            continue

        if not pid_obj.para_d:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            para_df = pd.DataFrame(pid_obj.para_d)
            filt_psv_0 = para_df.st_mode.str.contains('SPONT')
            filt_psv_1 = para_df.st_mode.str.contains('CPAP')
            filt_psv_2 = para_df.st_mode.str.contains('APNEA VENTILATION')
            # psv_len = para_df[filt_psv_0 | filt_psv_1
            #                   | filt_psv_2].ind.tolist()[-1]
            psv_df = para_df[filt_psv_0 | filt_psv_1 | filt_psv_2]
            psv_len = psv_df.ind.tolist()[-1] if not psv_df.empty else 0
            psv_len = round(psv_len / 60, 2)
            result_d = {'pid': pid, 'end': pid_obj.end_i, 'psv': psv_len}
            result_l.append(result_d)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))

    df = pd.DataFrame(result_l)
    pd.DataFrame.to_csv(df, Path.cwd() / 'psv_range_2.csv')


def PatientGen(gp, pid):
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
    return pid_obj


if __name__ == '__main__':
    main()