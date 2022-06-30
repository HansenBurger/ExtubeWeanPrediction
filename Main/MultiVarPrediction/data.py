import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.MLD.algorithm import LogisiticReg, RandomForest, SupportVector, XGBoosterClassify


class Basic():
    def __init__(self) -> None:
        pass


class StaticData(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__algo_set = {
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
                        0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                        0.35, 0.4, 0.45, 0.5
                    ],
                    'n_estimators':
                    range(100, 6000, 100),
                    'min_child_weight':
                    [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                    'gamma': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree':
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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

    @property
    def algo_set(self):
        return self.__algo_set