import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.KFoldPredict import KFoldMain
from Classes.ORM.expr import LabExtube, LabWean, PatientInfo
from Classes.FeatureProcess import FeatureLoader, FeatureProcess
from Classes.Func.KitTools import ConfigRead, SaveGen, measure
from Classes.MLD.algorithm import LogisiticReg, RandomForest, SupportVector, XGBoosterClassify

p_name = 'MultiInd'
mode_s = ['Extube_SumP12_Nad']

algorithm_set = {
    # 'LogReg': {
    #     'class': LogisiticReg,
    #     'split': 5,
    #     's_param': {
    #         'C': [0.001, 0.1, 1, 100, 1000],
    #         'penalty': ['l2', 'elasticnet'],
    #         'solver': ['liblinear', 'lbfgs'],
    #         'max_iter': [1, 10, 100, 1000, 2000]
    #     },
    #     'eval_set': False,
    #     'param_init': {},
    #     'param_deduce': {},
    #     're_select': False
    # },
    # 'RF': {
    #     'class': RandomForest,
    #     'split': 5,
    #     's_param': {
    #         'max_depth': range(10, 50, 3),
    #         'penalty': ['l1', 'l2'],
    #         'max_iter': [100, 500, 1000, 2000]
    #     },
    #     'eval_set': False,
    #     'param_init': {},
    #     'param_deduce': {},
    #     're_select': False
    # },
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
            'silent': True,
            'nthread': 4,
            'seed': 0
        },
        'param_deduce': {},
        're_select': True
    }
}


def main(mode_name):
    data_rot = Path(ConfigRead('VarData', mode_name))
    s_f_fold = SaveGen(Path(ConfigRead('ResultSave', 'Form')),
                       '-'.join([p_name, mode_name]))
    s_g_fold = SaveGen(Path(ConfigRead('ResultSave', 'Graph')),
                       '-'.join([p_name, mode_name]))

    data_var = FeatureLoader(data_rot).VarFeatLoad()
    data_group = GetGroupByICU(data_var)

    for group_, data_ in data_group.items():
        save_p_f = s_f_fold / group_
        save_p_f.mkdir(parents=True, exist_ok=True)
        save_p_g = s_g_fold / group_
        save_p_g.mkdir(parents=True, exist_ok=True)

        feature_p = FeatureProcess(data_, 'end', save_p_f)
        feature_p.FeatPerformance(
            data_.columns.drop(['pid', 'icu', 'end']).tolist())
        data_slt = feature_p.DataSelect(0.05, 0.1, 0.4, 0.2)

        if feature_p.feat.empty:
            print('{0} lack valid data'.format(group_))
            continue

        for k, v in algorithm_set.items():
            save_path = save_p_g / k
            save_path.mkdir(parents=True, exist_ok=True)
            model_p = KFoldMain(v['class'], v['split'])
            model_p.DataSetBuild(data_slt, 'end')
            model_p.ParamSelectRand(v['s_param'], v['eval_set'])
            model_p.CrossValidate(v['param_init'], v['param_deduce'],
                                  v['re_select'])
            model_p.ResultGenerate(save_path)


def GetGroupByICU(data_in: any) -> dict:
    group_s = {
        'TOT': data_in,
        'QC': data_in.loc[~data_in.icu.str.contains('xs')],
        'XS': data_in.loc[data_in.icu.str.contains('xs')]
    }

    for icu in data_in.icu.unique():
        group_s[icu] = data_in.loc[data_in.icu == icu]

    return group_s


# def KFoldCV(data_, save_path: Path):
#     pass

if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)