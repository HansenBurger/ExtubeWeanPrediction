import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Domain.layer_p import PatientVar
from Classes.Func.KitTools import PathVerify
from Classes.MLD.processfunc import DataSetProcess
from Classes.MLD.balancefunc import BalanSMOTE
from Classes.MLD.algorithm import LogisiticReg


class Basic():
    def __init__(self) -> None:
        pass

    def __TableLoad(self, load_path: Path):
        obj_s = []
        load_path = PathVerify(load_path)
        for file in load_path.iterdir():
            if not file.is_file():
                pass
            elif file.suffix == '.csv':
                obj = PatientVar()
                info_ = file.name.split('_')
                obj.pid = info_[0]
                obj.end = info_[1]
                obj.icu = info_[2]
                obj.data = pd.read_csv(file, index_col='method')
                obj_s.append(obj)

        return obj_s


class FeatureLoader(Basic):
    def __init__(self, local_p: Path):
        super().__init__()
        self.__samples = self._Basic__TableLoad(local_p)

    @property
    def samples(self):
        return self.__samples

    def VarFeatLoad(self, met_s: list = [], ind_s: list = []) -> pd.DataFrame:

        df_var = pd.DataFrame()
        met_s = met_s if met_s else self.__samples[0].data.index.to_list()
        ind_s = ind_s if ind_s else self.__samples[0].data.columns.to_list()

        df_var['pid'] = [samp.pid for samp in self.__samples]
        df_var['end'] = [samp.end for samp in self.__samples]
        df_var['icu'] = [samp.icu for samp in self.__samples]

        for met in met_s:
            for ind in ind_s:
                col_name = met + '-' + ind
                df_var[col_name] = [
                    samp.data.loc[met, ind] for samp in self.__samples
                ]

        return df_var

    def LabFeatCombine(self, src_0: any, src_1: any) -> pd.DataFrame:
        join_info = {
            'dest': src_0,
            'on': src_0.pid == src_1.pid,
            'attr': 'pinfo'
        }
        col_order = [src_1.pid]
        col_query = [src_1, src_0.age, src_0.sex, src_0.bmi]
        cond_pid = src_1.pid.in_([samp.pid for samp in self.__samples])

        que_l = src_1.select(*col_query).join(
            **join_info).where(cond_pid).order_by(*col_order)

        df_que = pd.DataFrame(list(que_l.dicts()))

        return df_que


class FeatureFilter(Basic):
    def __init__(self, data_: pd.DataFrame) -> None:
        super().__init__()
        self.__feat_col = [
            'met', 'end', 'P', 'AUC', 'LogReg', 'LogRegDiff', 'rs_0', 'size_0',
            'rs_1', 'size_1'
        ]
        self.__data = data_
        self.__feat = pd.DataFrame(dict().fromkeys(self.__feat_col))
