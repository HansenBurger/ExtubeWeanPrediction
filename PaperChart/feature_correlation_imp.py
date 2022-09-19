import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.Func.CalculatePart import PerfomAssess

p_name = 'FeatCorrelationImp'
mode_s = ["Extube_60_SumP12_Nad"]

json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)

order_st = {
    'GA': {
        'name': 'Group A',
        'mets': ['ave', 'med', 'qua', 'tqua', 'std', 'cv'],
        'inds': ['pip', 'rr', 'v_t', 've', 'rsbi']
    },
    'GB': {
        'name': 'Group B',
        'mets': ['ave', 'med', 'qua', 'tqua', 'std', 'cv'],
        'inds': ['wob', 'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t']
    },
    'GC': {
        'name': 'Group C',
        'mets':
        ['sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp', 'fuzz', 'ac', 'dc'],
        'inds': ['pip', 'rr', 'v_t', 've', 'rsbi']
    },
    'GD': {
        'name': 'Group D',
        'mets':
        ['sd1', 'sd2', 'pi', 'gi', 'si', 'app', 'samp', 'fuzz', 'ac', 'dc'],
        'inds': ['wob', 'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t']
    },
}

group_st = {
    'raw': {
        'name': 'a',
        'row_m': [],
        'col_m': [],
        'cmp': 'coolwarm'
    },
    'sub_0': {
        'name': 'b',
        'row_m': ['ave', 'med', 'qua', 'tqua'],
        'col_m': ['ave', 'med', 'qua', 'tqua'],
        'cmp': 'coolwarm'
    },
    'sub_1': {
        'name': 'c',
        'row_m': ['cv', 'std', 'sd1', 'sd2'],
        'col_m': ['cv', 'std', 'sd1', 'sd2'],
        'cmp': 'OrRd'
    },
    'sub_2': {
        'name': 'd',
        'row_m': ['std', 'cv', 'sd1', 'sd2'],
        'col_m': ['fuzz', 'ac', 'dc'],
        'cmp': 'coolwarm'
    },
    'sub_3': {
        'name': 'e',
        'row_m': ['app', 'samp', 'fuzz', 'ac', 'dc'],
        'col_m': ['app', 'samp', 'fuzz', 'ac', 'dc'],
        'cmp': 'coolwarm'
    }
}


class Basic():
    def __init__(self) -> None:
        pass


class FeatCorrMap(Basic):
    def __init__(self, mode_n: str) -> None:
        super().__init__()
        self.__ind_d = ConfigRead('RespVar', 'Indicators', json_loc)
        self.__met_d = ConfigRead('RespVar', 'Methods', json_loc)
        self.__load_p = Path(ConfigRead('Source', mode_n, json_loc))
        self.__corr_s = self.__GetCorrMap()
        self.__save_p = s_f_fold / mode_n
        self.__save_p.mkdir(parents=True, exist_ok=True)

    @property
    def corr_s(self):
        return self.__corr_s

    def __ImpFiltTable(self, df: pd.DataFrame) -> pd.DataFrame:
        imp_filt = df.abs() > 0.6
        df_imp = df[imp_filt]
        df_imp = df_imp.fillna(0)
        return df_imp

    def __GetFeatOrder(self) -> list:
        feat_order = []
        for v in order_st.values():
            g_feat_order = [[
                '$' + self.__met_d[m] + '-' + self.__ind_d[i] + '$'
                for i in v['inds']
            ] for m in v['mets']]
            g_feat_order = [x for l in g_feat_order for x in l]
            feat_order.extend(g_feat_order)
        return feat_order

    def __GetCorrMap(self) -> pd.DataFrame:
        local_df_s = []
        l_ex_end = []
        for f in self.__load_p.iterdir():
            if not f.is_file() or f.suffix != '.csv':
                continue
            else:
                l_ex_end.append(int(f.name.split('_')[1]))
                df = pd.read_csv(f, index_col='method')
                local_df_s.append(df)

        tot_d = {}
        for m_k, m_v in self.__met_d.items():
            for i_k, i_v in self.__ind_d.items():
                col_n = '$' + m_v + '-' + i_v + '$'
                feat_v = [df.loc[m_k, i_k] for df in local_df_s]
                feat_p, _, _ = PerfomAssess(l_ex_end, feat_v).PValueAssess()
                if feat_p >= 0.05:
                    continue
                else:
                    tot_d[col_n] = feat_v
        tot_df = pd.DataFrame(tot_d)
        corr_s = tot_df.corr(method='spearman')
        #corr_s = self.__ImpFiltTable(corr_s)
        corr_s_i = [i for i in self.__GetFeatOrder() if i in corr_s.index]
        corr_s = corr_s.reindex(corr_s_i)
        corr_s = corr_s[corr_s.index.to_list()]
        return corr_s

    def __AnnotateHeatMap(self, ax: any, name: str, row_m: list, col_m: list,
                          **kwargs) -> None:
        df = self.__corr_s.copy()
        x_r = max(ax.get_xlim()) / self.__corr_s.shape[1]
        y_r = max(ax.get_ylim()) / self.__corr_s.shape[0]
        row_m = [self.__met_d[i].split('(')[0] for i in row_m]
        col_m = [self.__met_d[i].split('(')[0] for i in col_m]
        row_s = df[df.index.str.contains('|'.join(row_m),
                                         regex=True)].index.tolist()
        col_s = df[df.index.str.contains('|'.join(col_m),
                                         regex=True)].index.tolist()
        x = df.index.tolist().index(col_s[0]) * x_r
        y = df.index.tolist().index(row_s[0]) * y_r
        w, h = len(col_s) * x_r, len(row_s) * y_r
        line_st = dict(fill=False, ec='k', lw=7, ls='--')
        fill_st = dict(fill=True, fc='whitesmoke', alpha=0.7)
        ax.add_patch(Rectangle((x, y), w, h, **fill_st))
        ax.add_patch(Rectangle((x, y), w, h, **line_st))

        x_c, y_c = x + 0.5 * w, y + 0.5 * h
        label_st = dict(fontsize=60, fontstyle='italic')
        ax.annotate(name, xy=(x_c, y_c), ha='center', va='center', **label_st)

    def __HeatMapPlot(self,
                      ax: any,
                      cmp: str,
                      name: str,
                      row_m: list = [],
                      col_m: list = [],
                      **kwargs) -> None:
        sns.reset_orig()
        color_st = [
            'royalblue', 'cornflowerblue', 'lightsteelblue', 'darksalmon',
            'coral', 'orangered'
        ]
        cmap = (mpl.colors.ListedColormap(color_st))
        bounds = [-1.0, -0.8, -0.6, 0.0, 0.6, 0.8, 1.0]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        if not row_m and not col_m:
            sns.heatmap(self.__corr_s, cmap=cmap, norm=norm, ax=ax, **kwargs)
        else:
            df = self.__corr_s.copy()
            row_m = [self.__met_d[i].split('(')[0] for i in row_m]
            col_m = [self.__met_d[i].split('(')[0] for i in col_m]
            row_f = df.index.str.contains('|'.join(row_m))
            col_s = df[df.index.str.contains('|'.join(col_m))].index.tolist()
            df = df[row_f]
            df = df.loc[:, col_s]
            sns.heatmap(df, cmap=cmap, norm=norm, ax=ax, **kwargs)
        ax.set_title('(' + name + ')',
                     loc='left',
                     fontsize=25,
                     fontstyle='italic')
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=6)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=6)

    def __SinglePlot(self, dim_st: tuple, save_n: str, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=dim_st, dpi=300)

        self.__HeatMapPlot(ax, linewidth=.1, **kwargs)
        fig.tight_layout()
        fig.savefig(self.__save_p / (save_n + '.png'))

    def Main(self):

        fig = plt.figure(figsize=(44, 20), dpi=300)
        gs_tot = fig.add_gridspec(1, 1)
        gs_0 = gs_tot[0].subgridspec(1, 1)

        ax_a = fig.add_subplot(gs_0[0, 0])
        self.__HeatMapPlot(ax_a, linewidth=.1, **group_st['raw'])
        self.__SinglePlot((22, 20), 'corr_a', **group_st['raw'])


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = FeatCorrMap(mode_n=mode_)
        main_p.Main()
