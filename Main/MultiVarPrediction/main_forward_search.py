import sys
from pathlib import Path
from pandas import DataFrame, concat
from data import StaticData
from func import MultiModelPredict, ResultsSummary

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.KitTools import ConfigRead, SaveGen, measure
from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.FeatureProcess import FeatureLoader, DatasetGeneration
'''
Forward Search Experiment
:Check config data before experiment
'''

sys_args = sys.argv
arg_suffix = sys_args[1] if len(sys_args) > 1 else 'Default'
arg_st_n = sys_args[2] if len(sys_args) > 2 else 'default'
arg_way = sys_args[3] if len(sys_args) > 3 else 'var'
arg_stop = sys_args[4] if len(sys_args) > 4 else 0.95

p_name = 'ForwardSearch' + '_' + arg_suffix
pred_way = 'KFold'  # KFold | Norm
static = StaticData()
save_p = SaveGen(Path(ConfigRead('ResultSave', 'Mix')), p_name)
mode_s = ['Extube_SumP12_AorD-60', 'Extube_SumP12_All-60']


class ForwardSearch():
    def __init__(self, mode_name: str) -> None:
        self.__mode_n = mode_name
        self.__s_f_p = None
        self.__s_g_p = None
        self.__feat_ = None
        self.__data_ = None

    def __SaveGen(self) -> list:
        s_f_folder = save_p / self.__mode_n / 'Form'
        s_f_folder.mkdir(parents=True, exist_ok=True)
        s_g_folder = save_p / self.__mode_n / 'Graph'
        s_g_folder.mkdir(parents=True, exist_ok=True)
        return s_f_folder, s_g_folder

    def __FeatDataLoad(self, way: str, **FeatSelect):
        data_rot = Path(ConfigRead('VarData', self.__mode_n))
        load_p = FeatureLoader(data_rot, self.__s_f_p)
        data_var, feat_var = load_p.VarFeatsLoad(**static.var_set[arg_st_n],
                                                 save_n='VarFeats')
        data_lab, feat_lab = load_p.LabFeatsLoad(PatientInfo, LabExtube,
                                                 'LabFeats')
        if way == 'var':
            data_p = DatasetGeneration(data_var, feat_var, self.__s_f_p)
            data_p.FeatsSelect(**FeatSelect)
            data_p.DataSelect()
        elif way == 'lab':
            data_p = DatasetGeneration(data_lab, feat_lab, self.__s_f_p)
            data_p.FeatsSelect(p_max=0.9)
            data_p.DataSelect()
        elif way == 'all':
            p_0 = DatasetGeneration(data_var, feat_var)
            p_1 = DatasetGeneration(data_lab, feat_lab)
            p_0.FeatsSelect(**FeatSelect)
            p_1.FeatsSelect(p_max=0.9)
            data_in = concat([p_0.data, p_1.data], axis=1)
            feat_in = concat([p_0.feat, p_1.feat], axis=0)
            data_in = data_in.loc[:, ~data_in.columns.duplicated()].copy()
            data_p = DatasetGeneration(data_in, feat_in, self.__s_f_p)
            data_p.DataSelect()

        return data_p.data, data_p.feat

    def __ForwardSearch(self, stop_k: str = 's_auc', stop_v: float = 1.0):
        feat_boxes = [[]]
        for i in range(len(self.__feat_)):
            feat_include = feat_boxes[i]
            feat_remain = [
                f for f in self.__feat_.index if not f in feat_include
            ]
            round_perform = {}

            for j in feat_remain:
                feat_j = [j] + feat_include
                data_j = self.__data_.loc[:, ['end'] + feat_j]
                lr_best_que = MultiModelPredict(data_j, static.algo_set, 'end')
                lr_best_que.MultiModels('KFold', ['LR'], store_results=False)
                round_perform[j] = lr_best_que.model_result['LR'].loc['s_auc']

            best_j = list(
                dict(sorted(round_perform.items(),
                            key=lambda item: item[1])).keys())[-1]

            feat_i = [best_j] + feat_include
            data_i = self.__data_.loc[:, ['end'] + feat_i]
            feat_boxes.append(feat_i)

            tot_predict = MultiModelPredict(
                data_set=data_i,
                algo_set=static.algo_set,
                label_col='end',
                save_path=self.__s_g_p /
                (str(i).rjust(len(str(len(self.__feat_))), '0') + '-' +
                 best_j))
            tot_predict.MultiModels(pred_way)

            perfom_ = [
                v[stop_k] > stop_v for v in tot_predict.model_result.values()
            ]
            if True in perfom_:
                print('Meat best line, end of search !')
                break

    def __ResultsCollect(self):
        main_p = ResultsSummary(self.__s_g_p, list(static.algo_set.keys()))
        main_p.TrendPlot()
        main_p.BestDisplay()

    def Main(self):
        self.__s_f_p, self.__s_g_p = self.__SaveGen()
        self.__data_, self.__feat_ = self.__FeatDataLoad(arg_way)
        self.__ForwardSearch()
        self.__ResultsCollect()


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = ForwardSearch(mode_)
        main_p.Main()