import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import GetObjectDict
from Classes.Func.DiagramsGen import PlotMain
from Classes.TypesInstant import ResultStatistical


class basic():
    def __init__(self) -> None:
        pass


class VarResultsGen(basic):
    def __init__(self, patient):
        super().__init__()
        self.__pid = patient  # layer3 Patient class

    def __SaveNaming(self):
        pid = self.__pid.pid
        end_i = self.__pid.end_i
        rid = self.__pid.rid_s.zif.name.split('.')[0]
        save_n = '{0}_{1}_{2}'.format(pid, end_i, rid)
        return save_n

    def VarRsGen(self, methods_l):
        resp_l = self.__pid.resp_l
        res_p = ResultStatistical(resp_l)
        res_p.CountAggr(methods_l)
        self.__pid.result = res_p.rec

    def ParaTrendsPlot(self, folder, col_sel):
        save_n = self.__SaveNaming() + '_para'
        para_d = self.__pid.para_d
        df = pd.DataFrame(para_d)
        PlotMain(folder).MultiLineplot('ind', col_sel, df, save_n)

    def RespTrendsPlot(self, folder, col_sel):
        resp_l = self.__pid.resp_l
        save_n = self.__SaveNaming() + '_wave'
        wid_l = [i.wid for i in resp_l]
        stl_l = [sum(wid_l[0:i]) for i in range(1, len(wid_l) + 1)]
        df = pd.DataFrame([GetObjectDict(i) for i in resp_l])
        df['ind'] = stl_l
        PlotMain(folder).MultiLineplot('ind', col_sel, df, save_n)

    def TensorStorage(self, folder):
        save_n = self.__SaveNaming()

        var_rs = self.__pid.result
        var_sl = [var_rs.td, var_rs.hra, var_rs.hrv, var_rs.ent, var_rs.prsa]
        var_sl = [GetObjectDict(i) for i in var_sl]
        var_sd = {}
        for i in var_sl:
            var_sd.update(i)
        var_save = []
        for k, v in var_sd.items():
            dict_ = {'method': k}
            dict_.update(GetObjectDict(v))
            var_save.append(dict_)
        df = pd.DataFrame(var_save).set_index(['method'])
        pd.DataFrame.to_csv(df, folder / (save_n + '.csv'))