import sys
import numpy as np
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
    def __init__(self, patient: any, methods: list = []):
        '''
        :para: patient: layer_p Patient classs
        :para: inds: resp parameters requested for outlierwip
        :para: mets: var methods requested
        '''
        super().__init__()
        self.__pid = patient
        self.__ind_s = [
            'pip', 'peep', 'rr', 'v_t_i', 've', 'rsbi', 'mp_jb_d', 'mp_jb_t',
            'mp_jm_d', 'mp_jl_d', 'mp_jm_t', 'mp_jl_t'
        ]
        self.__met_s = ['td', 'hra', 'hrv', 'ent', 'prsa'
                        ] if not methods else methods

    def __SaveNaming(self) -> str:
        pid = self.__pid.pid
        end_i = self.__pid.end_i
        icu = self.__pid.icu
        rid = self.__pid.rid_s.zif.name.split('.')[0]
        save_n = '{0}_{1}_{2}_{3}'.format(pid, end_i, icu, rid)
        return save_n

    def __OutliersWipe(self, max_d_st: int = 3) -> None:
        df = pd.DataFrame({})
        resp_l = self.__pid.resp_l
        outlier = lambda arr, max_d: (arr - np.mean(arr)) > max_d * np.std(arr)
        for ind in self.__ind_s:
            df[ind] = [getattr(resp, ind) for resp in resp_l]
            df[ind + '_val'] = ~outlier(df[ind], max_d_st)
        df_val = df[df[[ind + '_val' for ind in self.__ind_s]].all(axis=1)]
        self.__pid.resp_l = [resp_l[i] for i in df_val.index]

    def VarRsGen(self, scale_st:dict, **kwargs) -> None:
        self.__OutliersWipe()
        resp_l = self.__pid.resp_l
        res_p = ResultStatistical(resp_l, **scale_st)
        res_p.CountAggr(self.__met_s, **kwargs)

        #TODO empty verify
        # for met in self.__met_s:
        #     met_var = getattr(res_p.rec, met)
        #     for sub in list(GetObjectDict(met_var).keys()):
        #         met_sub_var = getattr(met_var, sub)
        #         for ind in list(GetObjectDict(met_sub_var).keys()):
        #             met_sub_ind_var = getattr(met_sub_var, ind)
        #             if met_sub_ind_var and not np.isnan(met_sub_ind_var):
        #                 continue
        #             else:
        #                 self.__pid.result = None
        #                 return

        self.__pid.result = res_p.rec
        return res_p.prsa_dist

    def ParaTrendsPlot(self, folder: Path, col_sel: list) -> None:
        save_n = self.__SaveNaming() + '_para'
        para_d = self.__pid.para_d
        df = pd.DataFrame(para_d)
        PlotMain(folder).MultiLineplot('ind', col_sel, df, save_n)

    def RespTrendsPlot(self,
                       folder: Path,
                       col_sel: list,
                       dtype: any = float) -> None:
        resp_l = self.__pid.resp_l
        save_n = self.__SaveNaming() + '_wave'
        wid_l = [i.wid for i in resp_l]
        stl_l = [sum(wid_l[0:i]) for i in range(1, len(wid_l) + 1)]
        df = pd.DataFrame([GetObjectDict(i) for i in resp_l])
        df['ind'] = stl_l
        PlotMain(folder).MultiLineplot('ind', col_sel, df, save_n)

    def TensorStorage(self, folder: Path, dtype: any = float) -> None:
        save_n = self.__SaveNaming()

        var_rs = self.__pid.result
        var_sl = [getattr(var_rs, i) for i in self.__met_s]
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