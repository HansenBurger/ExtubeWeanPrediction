import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from func import LocInit, TableQuery, PidVarCount, VarStatistics, VarInfoCollect
from Classes.Func.KitTools import ConfigRead, SaveGen, measure

p_name = 'VarAnalysis_60min'
mode_s = [
    'Extube_PSV_Nad', 'Extube_SumP12_Nad', 'Wean_PSV_Nad', 'Wean_SumP12_Nad'
]
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), p_name)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)


@measure
def main(mode_: str, t_set: int):
    LocInit(s_f_fold, s_g_fold, mode_)
    TableQuery(mode_)
    PidVarCount(t_set)
    VarInfoCollect()
    VarStatistics()


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_, 3600)