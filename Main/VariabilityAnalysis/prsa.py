import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from func import LocInit, TableQuery, PRSAVarCount, VarStatistics, VarInfoCollect, PRSAStatistics
from Classes.Func.KitTools import ConfigRead, SaveGen, measure

p_name = 'PRSA_60_min'
mode_s = ['Extube_SumP12_Nad']
s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')), p_name)
s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)

T_exp_st = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
s_exp_st = [2, 4, 6, 8, 10, 12, 14]

exp_st_s = []
for i in range(len(T_exp_st)):
    for j in range(len(s_exp_st)):
        exp_st = {'L': 120, 'F': 2.0}
        exp_st.update({'T': T_exp_st[i], 's': s_exp_st[j]})
        exp_st_s.append(exp_st)


@measure
def VarAnalysis(mode_: str, t_set: int):
    LocInit(s_f_fold, s_g_fold, mode_)
    TableQuery(mode_, (48, 2160))
    PRSAVarCount(t_set, exp_st_s)
    PRSAStatistics(T_exp_st, s_exp_st)


if __name__ == '__main__':
    for mode_ in mode_s:
        VarAnalysis(mode_, int(3600))
