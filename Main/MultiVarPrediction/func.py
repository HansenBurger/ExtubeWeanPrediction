import sys
from pathlib import Path
from pandas import DataFrame

sys.path.append(str(Path.cwd()))

from MLBasic.KFoldPredictor import KFoldMain


class Basic():
    def __init__(self) -> None:
        pass


class KFoldCossValid(Basic):
    def __init__(self,
                 data_set: DataFrame,
                 algo_set: dict,
                 save_path: Path = Path.cwd()):
        super().__init__()
        self.__data_set = data_set
        self.__algo_set = algo_set
        self.__save_path = save_path
        self.__ave_result = None

    @property
    def ave_result(self):
        return self.__ave_result

    def MultiKFold(self,
                   model_names: list = ['LR', 'RF', 'SVM', 'XGB'],
                   store_results: bool = True):
        self.__ave_result = {}
        for model in model_names:
            st = self.__algo_set[model]
            if not sum(self.__data_set.end.value_counts() < st['split']) == 0:
                continue
            save_path = self.__save_path / model

            model_p = KFoldMain(st['class'], st['split'])
            model_p.DataSetBuild(self.__data_set, 'end')
            model_p.ParamSelectRand(st['s_param'], st['eval_set'])
            model_p.CrossValidate(st['param_init'], st['param_deduce'],
                                  st['re_select'])
            df_rs = model_p.ResultGenerate(store_results, save_path)
            self.__ave_result[model] = df_rs.loc['ave', :]
