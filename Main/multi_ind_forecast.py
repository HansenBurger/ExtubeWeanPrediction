import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.FeatureProcess import FeatureLoader, FeatureProcess
from Classes.Func.KitTools import ConfigRead, SaveGen, measure

p_name = 'MultiInd'
mode_s = ['Extube_PSV_Nad', 'Extube_SumP12_Nad']


def main(mode_name):
    data_rot = Path(ConfigRead('VarData', mode_name))
    s_f_fold = SaveGen(
        Path(ConfigRead('ResultSave', 'Form')) / p_name, mode_name)
    s_g_fold = SaveGen(
        Path(ConfigRead('ResultSave', 'Graph')) / p_name, mode_name)

    data_var = FeatureLoader(data_rot).VarFeatLoad()
    data_group = GetGroupByICU(data_var)

    for data_ in data_group:
        pass


def GetGroupByICU(data_in: any) -> dict:
    group_s = {
        'TOT': data_in,
        'QC': data_in.loc[~data_in.icu.str.contains('xs')],
        'XS': data_in.loc[data_in.icu.str.contains('xs')]
    }

    for icu in data_in.icu.unique():
        group_s[icu] = data_in.loc[data_in.icu == icu]

    return group_s


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)