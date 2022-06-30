import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from func import LocInit, TableQuery, PidVarCount, VarStatistics, VarInfoCollect
from Classes.Func.KitTools import ConfigRead, SaveGen, measure

p_name = sys.argv[2]
mode_s = ['Extube_PSV_Nad', 'Extube_SumP12_Nad']
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), p_name)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)


@measure
def VarAnalysis(mode_: str, t_set: int):
    LocInit(s_f_fold, s_g_fold, mode_)
    TableQuery(mode_, (48, 2160))
    PidVarCount(t_set)
    VarInfoCollect()
    VarStatistics()


if __name__ == '__main__':
    for mode_ in mode_s:
        VarAnalysis(mode_, sys.argv[1])