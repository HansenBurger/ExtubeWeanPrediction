import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Domain.layer_p import PatientVar
from Classes.Func.CalculatePart import PerfomAssess
from Classes.Func.KitTools import SaveGen, PathVerify
from Classes.ORM.expr import PatientInfo, LabExtube, LabWean
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12, WeanPSV, WeanSumP12


class Basic():
    def __init__(self) -> None:
        pass


class DataLoader(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__samples = []

    @property
    def samples(self):
        return self.__samples

    def SampleInit(self, local_p: Path):
        local_p = PathVerify(local_p)
        for p in local_p.iterdir():
            if not p.is_file():
                pass
            elif p.suffix == '.csv':
                p_obj = PatientVar()
                p_info = p.name.split('_')
                p_obj.pid = p_info[0]
                p_obj.end = p_info[1]
                p_obj.icu = p_info[2]
                p_obj.data = pd.read_csv(p, index_col='method')
                self.__samples.append(p_obj)

    def VarFeatLoad(self,
                    met_s: list(str) = [],
                    ind_s: list(str) = []) -> pd.DataFrame:

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