import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from func import LocInit, TableQuery, PidVarCount, VarStatistics, VarInfoCollect
from Classes.Func.KitTools import ConfigRead, SaveGen, measure

p_set = sys.argv[2] if len(sys.argv) > 2 else "Test"
t_set = int(sys.argv[1]) if len(sys.argv) > 2 else 3600
s_set = {
    'ind_stride': float(sys.argv[3]),
    'ind_range': float(sys.argv[3]),
    'var_stride': 0,
    'var_range': 3600
}
# p_set, t_set, s_set = "Test", 3600, {
#     'ind_stride': 4,
#     'ind_range': 4,
#     'var_stride': 0,
#     'var_range': 3600
# }

mode_s = ['Extube_SumP12_Nad']
s_fold = SaveGen(Path(ConfigRead('ResultSave', 'Mix')), p_set)
s_f_fold = s_fold / "Form"
s_g_fold = s_fold / "Chart"
s_f_fold.mkdir(parents=True, exist_ok=True)
s_g_fold.mkdir(parents=True, exist_ok=True)


@measure
def VarAnalysis(mode_: str, t_set: int, s_set: float):
    LocInit(s_f_fold, s_g_fold, mode_)
    TableQuery(mode_, (48, 2160))
    PidVarCount(t_set, s_set)
    VarInfoCollect()
    VarStatistics()


if __name__ == '__main__':

    for mode_ in mode_s:
        VarAnalysis(mode_, t_set, s_set)