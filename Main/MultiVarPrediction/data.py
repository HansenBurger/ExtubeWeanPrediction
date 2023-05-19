import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from MLBasic.Classes.algorithm import LogisiticReg, RandomForest, SupportVector, XGBoosterClassify


class Basic():
    def __init__(self) -> None:
        pass


class StaticData(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__algo_set = {
            'LR': {
                'class': LogisiticReg,
                'split': 0.3,
                'k-split': 5,
                's_param': {
                    'C': [0.001, 0.1, 1, 100, 1000, 5000],
                    'penalty': ['l2'],
                    'solver': ['liblinear'],
                    'max_iter': [1, 10, 100, 1000, 2000, 5000]
                },
                'eval_set': False,
                'param_init': {},
                'param_deduce': {},
                're_select': False,
                'get_feat_imp': True
            },
            'RF': {
                'class': RandomForest,
                'split': 0.3,
                'k-split': 5,
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
                're_select': False,
                'get_feat_imp': True
            },
            'SVM': {
                'class': SupportVector,
                'split': 0.3,
                'k-split': 5,
                's_param': {
                    'C': [0.1, 1, 10, 100, 1000, 5000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                    'kernel': ['rbf'],
                    'probability': [True]
                },
                'eval_set': False,
                'param_init': {},
                'param_deduce': {},
                're_select': False,
                'get_feat_imp': False
            },
            'XGB': {
                'class': XGBoosterClassify,
                'split': 0.3,
                'k-split': 5,
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
                're_select': True,
                'get_feat_imp': True
            }
        }
        self.__var_set = {
            'basic_Nvar': {
                # Basic Inds + Basic Methods (Not only variability)
                'met_s': ['cv', 'std', 'ave', 'med', 'qua', 'tqua'],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            'basic_var': {
                # Basic Inds + Basic Methods (Only variability)
                'met_s': ['cv', 'std'],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            'basic_dis': {
                # Basic Inds + Inds Distribution
                'met_s': ['ave', 'med', 'qua', 'tqua'],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            'inds_Nvar': {
                # All Inds + Basic Methods (Not only variability)
                'met_s': ['cv', 'std', 'ave', 'med', 'qua', 'tqua'],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
                    'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'
                ]
            },
            'inds_var': {
                # All Inds + Basic Methods (Only variability)
                'met_s': ['cv', 'std'],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
                    'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'
                ]
            },
            'inds_dis': {
                # All Inds + Inds Distribution
                'met_s': ['ave', 'med', 'qua', 'tqua'],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
                    'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'
                ]
            },
            'mets_Nvar': {
                # Basic Inds + All Methods (Not only variability)
                'met_s': [],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            'mets_var': {
                # Basic Inds + All Methods (Only variability)
                'met_s': [
                    'cv', 'std', 'sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp',
                    'fuzz', 'ac', 'dc'
                ],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            'all_Nvar': {
                # All Inds + All Methods (Not only variability)
                'met_s': [],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
                    'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'
                ]
            },
            'all_var': {
                # All Inds + All Methods (Only variability)
                'met_s': [
                    'cv', 'std', 'sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp',
                    'fuzz', 'ac', 'dc'
                ],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
                    'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'
                ]
            },
            'exp_best': {
                # MP dynamic + All Methods (Not only variability)
                'met_s': [],
                'ind_s': [
                    'pip', 'rr', 'v_t', 've', 'rsbi', 'mp_jb_d', 'mp_jl_d', 'mp_jm_d'
                ]
            },
            'default': {
                'met_s': [],
                'ind_s': []
            }
        }
        self.__feat_cate = {
            # Basic ind + Basic method
            'group_a': {
                'met_s': ['ave', 'med', 'qua', 'tqua', 'std', 'cv'],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            # Comprehensive ind + Basic method
            'group_b': {
                'met_s': ['ave', 'med', 'qua', 'tqua', 'std', 'cv'],
                'ind_s': [
                    'mp_jb_d', 'mp_jb_t', 'mp_jl_d', 'mp_jm_d', 'mp_jl_t',
                    'mp_jm_t'
                ]
            },
            # Basic ind + Advanced method
            'group_c': {
                'met_s': [
                    'sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp', 'fuzz',
                    'ac', 'dc'
                ],
                'ind_s': ['pip', 'rr', 'v_t', 've', 'rsbi']
            },
            # Comprehensive ind + Advanced method
            'group_d': {
                'met_s': [
                    'sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp', 'fuzz',
                    'ac', 'dc'
                ],
                'ind_s': [
                    'mp_jb_d', 'mp_jb_t', 'mp_jl_d', 'mp_jm_d', 'mp_jl_t',
                    'mp_jm_t'
                ]
            }
        }

    @property
    def algo_set(self):
        return self.__algo_set

    @property
    def var_set(self):
        return self.__var_set

    @property
    def feat_cate(self):
        return self.__feat_cate