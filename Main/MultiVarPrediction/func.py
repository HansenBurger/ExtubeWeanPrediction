import sys
from pathlib import Path
from pandas import DataFrame

sys.path.append(str(Path.cwd()))

from MLBasic.Predictors import NormPredictor, KFoldPredictor


class Basic():
    def __init__(self) -> None:
        pass


class MultiModelPredict(Basic):
    def __init__(self,
                 data_set: DataFrame,
                 algo_set: dict,
                 label_col: str,
                 save_path: Path = Path.cwd()):
        super().__init__()
        self.__data_set = data_set
        self.__algo_set = algo_set
        self.__label_col = label_col
        self.__save_path = save_path
        self.__model_result = {}

    @property
    def model_result(self):
        return self.__model_result

    def __SplitValid(self, split_n: any):
        dist_count = self.__data_set[self.__label_col].value_counts()
        dist_filt = dist_count < split_n
        validation = sum(dist_filt) == 0

        return validation

    def Basic(self, m_st: dict, save_p: Path, store_results: bool = True):
        model_p = NormPredictor(m_st['class'])
        model_p.DataSetBuild(self.__data_set, self.__label_col, m_st['split'])
        model_p.ParamSelectRand(m_st['s_param'], m_st['eval_set'])
        model_p.Validate(m_st['param_init'], m_st['param_deduce'],
                         m_st['re_select'], m_st['get_feat_imp'])
        df_rs = model_p.ResultGenerate(store_results, save_p)
        return df_rs

    def KFold(self, m_st: dict, save_p: Path, store_results: bool = True):
        if not self.__SplitValid(m_st['k-split']):
            return None
        model_p = KFoldPredictor(m_st['class'], m_st['k-split'])
        model_p.DataSetBuild(self.__data_set, self.__label_col)
        model_p.ParamSelectRand(m_st['s_param'], m_st['eval_set'])
        model_p.CrossValidate(m_st['param_init'], m_st['param_deduce'],
                              m_st['re_select'], m_st['get_feat_imp'])
        df_rs = model_p.ResultGenerate(store_results, save_p)
        df_ave = df_rs.loc['ave', :]
        return df_ave

    def MultiModels(self,
                    pred_way: str,
                    model_names: list = ['LR', 'RF', 'SVM', 'XGB'],
                    store_results: bool = True):
        for model in model_names:
            m_st = self.__algo_set[model]
            save_path = self.__save_path / model
            if pred_way == 'Norm':
                rs = self.Basic(m_st, save_path, store_results)
                self.__model_result[model] = rs
            elif pred_way == 'KFold':
                rs = self.KFold(m_st, save_path, store_results)
                self.__model_result[model] = rs
