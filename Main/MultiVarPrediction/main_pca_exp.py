import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from data import StaticData
from func import MultiModelPredict, ResultsSummary

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.FeatureProcess import FeatureLoader, DatasetGeneration

p_name = 'PCAExperiment'
pred_way = 'KFold'

static = StaticData()
save_p = SaveGen(Path(ConfigRead('ResultSave', 'Mix')), p_name)
mode_s = ['Extube_SumP12_Nad-60']

feats_slt = [
    'ac-mp_jl_d', 'gi-wob', 'dc-pip', 'tqua-rr', 'std-pip', 'cv-pip', 'ac-v_t',
    'std-mp_jl_t', 'cv-mp_jl_d', 'sd2-mp_jl_t', 'sd1-v_t'
]

feats_gps = {
    'gp_0': ['ac-mp_jl_d', 'dc-pip'],
    'gp_1': ['std-pip', 'cv-pip', 'std-mp_jl_t', 'cv-mp_jl_d', 'sd2-mp_jl_t'],
    'gp_2': ['ac-v_t', 'sd1-v_t']
}


class Basic():
    def __init__(self, mode_name: str) -> None:
        self.__mode_n = mode_name

    def __SaveGen(self) -> list:
        s_f_folder = save_p / self.__mode_n / 'Form'
        s_f_folder.mkdir(parents=True, exist_ok=True)
        s_g_folder = save_p / self.__mode_n / 'Graph'
        s_g_folder.mkdir(parents=True, exist_ok=True)
        return s_f_folder, s_g_folder


class PCAExp(Basic):
    def __init__(self, mode_name: str) -> None:
        super().__init__(mode_name)
        self.__s_f_p = None
        self.__s_g_p = None
        self.__data_ = None

    def __PCAProcess(self, tot_data: pd.DataFrame, gp_feat_d: dict) -> None:
        gp_pca_d = {}
        extra_data = tot_data.copy()
        for k, v in gp_feat_d.items():
            pca = PCA(n_components=1)
            gp_pca_d[k + '_pca'] = pca.fit_transform(tot_data.loc[:, v]).T[0]
            extra_data = extra_data.drop(columns=v)
        pca_data = pd.concat([extra_data, pd.DataFrame(gp_pca_d)], axis=1)
        return pca_data

    def __FeatDataLoad(self, **FeatSelect):
        data_rot = Path(ConfigRead('VarData', self._Basic__mode_n))
        load_p = FeatureLoader(data_rot, self.__s_f_p)
        data_var, feat_var = load_p.VarFeatsLoad(spec_=feats_slt,
                                                 save_n='VarFeats')
        _ = load_p.LabFeatsLoad(PatientInfo, LabExtube, 'LabFeats')
        data_p = DatasetGeneration(data_var, feat_var, self.__s_f_p)
        data_p.FeatsSelect(**FeatSelect)
        data_p.DataSelect()
        pca_data = self.__PCAProcess(data_p.data, feats_gps)
        return pca_data

    def __KFoldTest(self, data: pd.DataFrame, model: str = 'XGB'):
        tot_predict = MultiModelPredict(data_set=data,
                                        algo_set=static.algo_set,
                                        label_col='end',
                                        save_path=self.__s_g_p / model)
        tot_predict.MultiModels(pred_way, model_names=[model])

    def Main(self):
        self.__s_f_p, self.__s_g_p = self._Basic__SaveGen()
        self.__data_ = self.__FeatDataLoad()
        self.__KFoldTest(self.__data_)


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = PCAExp(mode_)
        main_p.Main()
