import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain, plt

ind_s = [
    'rr', 'v_t_i', 've', 'rsbi', 'wob', 'mp_jm_d', 'mp_jl_d', 'mp_jm_t',
    'mp_jl_t'
]


def GetOutliers(arr_: np.array, max_d: int = 3):
    dis_mean = arr_ - np.mean(arr_)
    outliers = dis_mean > max_d * np.std(arr_)
    return outliers


def RespValStatic(resp_l: list, save_loc: Path):
    df = pd.DataFrame()
    wid_l = [resp.wid for resp in resp_l]
    tot_len = round(sum(wid_l), 2)
    df['t_wid'] = wid_l
    df['t_ind'] = [sum(wid_l[0:i]) for i in range(len(wid_l))]
    p_plot = PlotMain(save_loc)
    for ind in ind_s:
        df[ind] = [getattr(resp, ind) for resp in resp_l]
        df[ind + '_val'] = ~GetOutliers(df[ind])
        p_plot.lmplot('t_ind', ind, ind + '_val', df, 'dist_' + ind)
    df_val = df[df[[ind + '_val' for ind in ind_s]].all(axis=1)]
    resp_val_l = [resp_l[i] for i in df_val.index]

    df.to_csv(save_loc / 'resp_ind_all.csv', index=False)
    df_val.to_csv(save_loc / 'resp_ind_val.csv', index=False)
    val_len = round(sum(df_val.t_wid), 2)

    with open(save_loc / 'tot_{0},val_{1}.txt'.format(tot_len, val_len),
              'w') as f:
        pass

    return resp_val_l


def PoincareScatter(resp_l: list, save_loc: Path = Path.cwd()) -> None:
    def __Panglais(list_: list) -> np.array:
        a_0 = np.array(list_[:len(list_) - 1])
        a_1 = np.array(list_[1:])
        return a_0, a_1

    resp_df = pd.DataFrame()

    if not save_loc.is_dir():
        save_loc.mkdir(parents=True, exist_ok=True)

    for ind in ind_s:
        a_0, a_1 = __Panglais([getattr(resp, ind) for resp in resp_l])
        resp_df[ind + '_0'] = a_0
        resp_df[ind + '_1'] = a_1
        sns.reset_orig()
        sns.set_theme(style="whitegrid")
        plt.plot(a_0, a_1, 'bo')
        plt.title(ind)
        plt.tight_layout()
        plt.savefig(save_loc / (ind + '.png'))
        plt.close()

    resp_df.to_csv(save_loc / 'resp_data.csv', index=False)