import sys
from pathlib import Path
from pandas import DataFrame
from data import StaticData
from func import MultiModelPredict

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.KitTools import ConfigRead, SaveGen, measure
from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.FeatureProcess import FeatureLoader, DatasetGeneration
'''
Forward Search Experiment
:function :
'''

sys_args = sys.argv
arg_suffix = sys_args[1] if len(sys_args) > 1 else 'Default'
arg_pred = sys_args[2] if len(sys_args) > 2 else 'KFold'
arg_st_n = sys_args[3] if len(sys_args) > 3 else 'all_Nvar'
arg_stop = sys_args[4] if len(sys_args) > 4 else 0.95

p_name = 'ForwardSearch' + '_' + arg_suffix
pred_way = arg_pred  # KFold | Norm
static = StaticData()
save_p = SaveGen(Path(ConfigRead('ResultSave', 'Mix')), p_name)
mode_s = ['Extube_SumP12_Nad-30', 'Extube_SumP12_Nad-60']


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

    def __FeatDataLoad(self, **FeatSelect):
        data_rot = Path(ConfigRead('VarData', self.__mode_n))
        load_p = FeatureLoader(data_rot, self.__s_f_p)
        data_var, feat_var = load_p.VarFeatsLoad(**static.var_set[arg_st_n],
                                                 save_n='VarFeats')
        _ = load_p.LabFeatsLoad(PatientInfo, LabExtube, 'LabFeats')

        data_p = DatasetGeneration(data_var, feat_var, self.__s_f_p)
        data_p.FeatsSelect(**FeatSelect)
        data_p.DataSelect()
        return data_p.data, data_p.feat

    def __ForwardSearch(self, stop_auc: float = 1.0):
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

    def __ResultsSummary(self):
        pass

    def Main(self):
        self.__s_f_p, self.__s_g_p = self.__SaveGen()
        self.__data_, self.__feat_ = self.__FeatDataLoad()
        self.__ForwardSearch()


@measure
def main(mode_name: str):
    data_rot = Path(ConfigRead('VarData', mode_name))
    s_f_fold = save_p / mode_name / 'Form'
    s_f_fold.mkdir(parents=True, exist_ok=True)
    s_g_fold = save_p / mode_name / 'Graph'
    s_g_fold.mkdir(parents=True, exist_ok=True)

    load_p = FeatureLoader(data_rot, s_f_fold)
    data_var, feat_var = load_p.VarFeatsLoad(**static.var_set[arg_st_n],
                                             save_n='VarFeats')
    _ = load_p.LabFeatsLoad(PatientInfo, LabExtube, 'LabFeats')

    data_p = DatasetGeneration(data_var, feat_var, s_f_fold)
    data_p.FeatsSelect()
    data_p.DataSelect()
    data_tot, feat_tot = data_p.data, data_p.feat

    feat_boxes = [[]]
    perform_boxes = [[]]
    for i in range(len(feat_tot)):
        feat_include = feat_boxes[i]
        feat_remain = [f for f in feat_tot.index if not f in feat_include]
        round_perform = {}
        for j in feat_remain:
            feat_j = [j] + feat_include
            data_j = data_tot.loc[:, ['end'] + feat_j]
            lr_best_que = MultiModelPredict(data_j, static.algo_set, 'end')
            lr_best_que.MultiModels('KFold', ['LR'], store_results=False)
            round_perform[j] = lr_best_que.model_result['LR'].loc['s_auc']
        best_j = list(
            dict(sorted(round_perform.items(),
                        key=lambda item: item[1])).keys())[-1]

        feat_i = [best_j] + feat_include
        data_i = data_tot.loc[:, ['end'] + feat_i]
        feat_boxes.append(feat_i)
        tot_predict = MultiModelPredict(
            data_i, static.algo_set, 'end', s_g_fold /
            (str(i).rjust(len(str(len(feat_tot))), '0') + '-' + best_j))
        tot_predict.MultiModels(pred_way)
        perform_boxes.append(tot_predict.model_result)

    for model in ['LR', 'RF', 'SVM', 'XGB']:
        model_tot = DataFrame({})
        model_tot['sen'] = [i[model]['s_sen'] for i in perform_boxes[1:]]
        model_tot['spe'] = [i[model]['s_spe'] for i in perform_boxes[1:]]
        model_tot['acc'] = [i[model]['s_acc'] for i in perform_boxes[1:]]
        model_tot['auc'] = [i[model]['s_auc'] for i in perform_boxes[1:]]
        model_tot['f_1'] = [i[model]['s_f_1'] for i in perform_boxes[1:]]
        model_tot['ind'] = [i + 1 for i in model_tot.index.to_list()]
        p_plot = PlotMain(s_g_fold)
        p_plot.linesplot('ind', ['auc', 'acc', 'f_1', 'sen', 'spe'], model_tot,
                         model + '_Perform')


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = ForwardSearch(mode_)
        main_p.Main()