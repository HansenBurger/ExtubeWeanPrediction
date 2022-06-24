import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain

ind_s = [
    'rr', 'v_t_i', 've', 'rsbi', 'wob', 'mp_jm_d', 'mp_jl_d', 'mp_jm_t',
    'mp_jl_t'
]


def GetOutliers(arr_: np.array, max_d: int = 4):
    dis_mean = arr_ - np.mean(arr_)
    outliers = dis_mean > max_d * np.std(arr_)
    return outliers


def OutliersDel(self,
                array_1: np.array,
                array_2: np.array,
                max_deviation: int = 4) -> None:

    arr_1_outs = GetOutliers(array_1, max_deviation)
    arr_2_outs = GetOutliers(array_2, max_deviation)

    arrs_not_outs = ~(arr_1_outs | arr_2_outs)

    array_1 = array_1[arrs_not_outs]
    array_2 = array_2[arrs_not_outs]

    return array_1, array_2


def RespValStatic(resp_l: list):
    df = pd.DataFrame()
    wid_l = [resp.wid for resp in resp_l]
    df['t_ind'] = [sum(wid_l[0:i]) for i in range(len(wid_l))]
    p_plot = PlotMain()
    for ind in ind_s:
        df[ind] = [getattr(resp, ind) for resp in resp_l]
        df[ind + '_val'] = ~GetOutliers(df[ind])
        p_plot.lmplot('t_ind', ind, ind + '_val', df)
        a = 1
