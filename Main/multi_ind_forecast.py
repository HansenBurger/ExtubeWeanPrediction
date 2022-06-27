import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.KFoldPredict import KFoldMain
from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.Func.DiagramsGen import PlotMain
from Classes.FeatureProcess import FeatureLoader, FeatureProcess
from Classes.Func.KitTools import ConfigRead, SaveGen, measure
from Classes.MLD.algorithm import LogisiticReg, RandomForest, SupportVector, XGBoosterClassify

p_name = 'MultiInd-30min&60min-STD3'
mode_s = [
    'Extube_PSV_Nad-30', 'Extube_SumP12_Nad-30', 'Extube_PSV_Nad-60',
    'Extube_SumP12_Nad-60'
]
# feats_demand = {'All_in': {'p_max': 2}}
# feats_demand = {
#     # type_1: Only the Imp P-Feature
#     'Type1': {},
#     # type_2: POS/NEG GreatImp Feature
#     'Type2': {
#         'diff_min': 0.1
#     },
#     # type_3: POSImp Feature
#     'Type3': {
#         'auc_min': 0.5,
#         'diff_min': 0.01
#     },
#     # type_4: POS GreatImp Feature
#     'Type4': {
#         'auc_min': 0.5,
#         'diff_min': 0.1
#     }
# }

s_f_path = SaveGen(Path(ConfigRead('ResultSave', 'Form')), p_name)
s_g_path = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)

algorithm_set = {
    'LR': {
        'class': LogisiticReg,
        'split': 5,
        's_param': {
            'C': [0.001, 0.1, 1, 100, 1000, 5000],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [1, 10, 100, 1000, 2000, 5000]
        },
        'eval_set': False,
        'param_init': {},
        'param_deduce': {},
        're_select': False
    },
    'RF': {
        'class': RandomForest,
        'split': 5,
        's_param': {
            'max_depth': range(10, 80, 5),
            'n_estimators': range(100, 2000, 100),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 4, 8, 16],
            'bootstrap': [True, False],
            'criterion': ['entropy']
        },
        'eval_set': False,
        'param_init': {},
        'param_deduce': {},
        're_select': False
    },
    'SVM': {
        'class': SupportVector,
        'split': 5,
        's_param': {
            'C': [0.1, 1, 10, 100, 1000, 5000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
            'kernel': ['rbf'],
            'probability': [True]
        },
        'eval_set': False,
        'param_init': {},
        'param_deduce': {},
        're_select': False
    },
    'XGB': {
        'class': XGBoosterClassify,
        'split': 5,
        's_param': {
            'booster': ["gbtree"],
            'learning_rate': [
                0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                0.45, 0.5
            ],
            'n_estimators':
            range(100, 6000, 100),
            'min_child_weight':
            [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'gamma': [0, 1e-1, 1, 5, 10, 20, 50, 100],
            'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'max_depth': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "scale_pos_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'reg_alpha': [1, 0.5, 0.1, 0.08]
        },
        'eval_set': True,
        'param_init': {
            'verbosity': 0,
            'nthread': 4,
            'seed': 0
        },
        'param_deduce': {},
        're_select': True
    }
}


@measure
def main(mode_name, filt_type: str = ''):
    data_rot = Path(ConfigRead('VarData', mode_name))
    s_f_fold = s_f_path / mode_name
    s_g_fold = s_g_path / mode_name

    load_p = FeatureLoader(data_rot)
    data_var = load_p.VarFeatLoad()
    data_que = load_p.LabFeatLoad(PatientInfo, LabExtube)
    # data_group = GetGroupByRMK([data_var, data_que])
    data_tot = {'Total': [data_var, data_que]}
    # data_group.update(data_tot)

    for group_, data_s in data_tot.items():

        save_p_f = s_f_fold / group_
        save_p_f.mkdir(parents=True, exist_ok=True)
        save_p_g = s_g_fold / group_
        save_p_g.mkdir(parents=True, exist_ok=True)

        info_col_n = ['pid', 'icu', 'end', 'rmk']

        feat_que_s = data_s[1].columns.drop(info_col_n).tolist()
        feat_que_p = FeatureProcess(data_s[1], 'end', save_p_f)
        feat_que_p.FeatPerformance(feat_que_s, 'LabFeats')

        feat_var_s = data_s[0].columns.drop(info_col_n).tolist()
        feat_var_p = FeatureProcess(data_s[0], 'end', save_p_f)
        feat_var_p.FeatPerformance(feat_var_s, 'VarFeats')
        data_slt = feat_var_p.DataSelect()
        feat_slt = feat_var_p.feat

        if data_slt.empty:
            print('{0} lack valid data'.format(group_))
            continue

        for k, v in algorithm_set.items():
            if not sum(data_slt.end.value_counts() < v['split']) == 0:
                continue
            dict_s = []
            save_path = save_p_g / k
            save_path.mkdir(parents=True, exist_ok=True)
            for ind in range(feat_slt.shape[0]):
                # if ind != feat_slt.shape[0] - 1:
                #     continue
                dict_ = {}
                feat_s = feat_slt.iloc[0:ind + 1].met.tolist()
                data_tmp = data_slt.loc[:, ['end'] + feat_s]
                model_p = KFoldMain(v['class'], v['split'])
                model_p.DataSetBuild(data_tmp, 'end')
                model_p.ParamSelectRand(v['s_param'], v['eval_set'])
                model_p.CrossValidate(v['param_init'], v['param_deduce'],
                                      v['re_select'])
                save_ind = save_path / str(ind)
                save_ind.mkdir(parents=True, exist_ok=True)
                model_p.ResultGenerate(save_ind)

                dict_['feat_n'] = ind + 1
                dict_['feats'] = ('|').join(feat_s)
                dict_['auc_mean'] = model_p.ave_result['auc']
                dict_['f1_mean'] = model_p.ave_result['f1']
                dict_['acc_mean'] = model_p.ave_result['r2']
                dict_['sens_mean'] = model_p.ave_result['sens']
                dict_['spec_mean'] = model_p.ave_result['spec']
                dict_s.append(dict_)

            df_tot = pd.DataFrame(dict_s)
            pd.DataFrame.to_csv(df_tot,
                                save_path / 'tot_performance.csv',
                                index=False)
            p_plot = PlotMain(save_path)
            p_plot.linesplot(
                'feat_n',
                ['auc_mean', 'f1_mean', 'acc_mean', 'sens_mean', 'spec_mean'],
                df_tot, 'performance')


def GetGroupByICU(data_in: list, excludings: list = []) -> dict:
    group_s = {
        'QC': data_in.loc[~data_in.icu.str.contains('xs')],
        'XS': data_in.loc[data_in.icu.str.contains('xs')]
    }

    for icu in data_in.icu.unique():
        group_data = data_in.loc[data_in.icu == icu]
        group_dist = len(group_data.end.unique())
        group_s[icu] = group_data if group_dist > 1 else None

    group_s = {
        k: v
        for k, v in group_s.items() if v is not None and not k in excludings
    }
    return group_s


def GetGroupByRMK(data_s: list, excludings: list = []) -> dict:
    group_s = {}

    for rmk in data_s[0].rmk.unique():
        group_data = data_s[0].loc[data_s[0].rmk == rmk]
        group_dist = len(group_data.end.unique())
        rmk = (',').join(rmk.split('/'))
        group_s[rmk] = group_data.index if group_dist > 1 else None

    group_s = {
        k: [data.iloc[v] for data in data_s]
        for k, v in group_s.items() if v is not None and not k in excludings
    }

    return group_s


if __name__ == '__main__':
    for mode_ in mode_s:
        print('round {0}'.format(mode_))
        main(mode_)