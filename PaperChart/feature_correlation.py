import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen

p_name = 'FeatCorrelation'
mode_s = ["Extube_60_SumP12_Nad"]

json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)

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

    def __GetCorrMap(self) -> pd.DataFrame:
        local_df_s = []
        for f in self.__load_p.iterdir():
            if not f.is_file() or f.suffix != '.csv':
                continue
            else:
                df = pd.read_csv(f, index_col='method')
                local_df_s.append(df)

        tot_d = {}
        for m_k, m_v in self.__met_d.items():
            for i_k, i_v in self.__ind_d.items():
                col_n = '$' + m_v + '-' + i_v + '$'
                tot_d[col_n] = [df.loc[m_k, i_k] for df in local_df_s]
        tot_df = pd.DataFrame(tot_d)
        corr_s = tot_df.corr(method='spearman')
        corr_s = self.__ImpFiltTable(corr_s)
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
        # circle_st_b = dict(fc='whitesmoke', alpha=0.7)
        # circle_st_e = dict(fill=False, ec='k', lw=7)
        # ax.add_patch(plt.Circle((x_c, y_c), radius=0.15 * w, **circle_st_b))
        # ax.add_patch(plt.Circle((x_c, y_c), radius=0.15 * w, **circle_st_e))
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
        if not row_m and not col_m:
            sns.heatmap(self.__corr_s, cmap=cmp, ax=ax, **kwargs)
        else:
            df = self.__corr_s.copy()
            row_m = [self.__met_d[i].split('(')[0] for i in row_m]
            col_m = [self.__met_d[i].split('(')[0] for i in col_m]
            row_f = df.index.str.contains('|'.join(row_m))
            col_s = df[df.index.str.contains('|'.join(col_m))].index.tolist()
            df = df[row_f]
            df = df.loc[:, col_s]
            sns.heatmap(df, cmap=cmp, ax=ax, **kwargs)
        ax.set_title('(' + name + ')',
                     loc='left',
                     fontsize=25,
                     fontstyle='italic')
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=6)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=6)

    def __SinglePlot(self, dim_st: tuple, save_n: str, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=dim_st)
        self.__HeatMapPlot(ax, linewidth=.1, **kwargs)
        fig.tight_layout()
        fig.savefig(self.__save_p / (save_n + '.png'))

    def Main(self):

        fig = plt.figure(figsize=(44, 20), dpi=300)
        gs_tot = fig.add_gridspec(1, 2)
        gs_0 = gs_tot[0].subgridspec(1, 1)
        gs_1 = gs_tot[1].subgridspec(2, 2)

        ax_a = fig.add_subplot(gs_0[0, 0])
        self.__HeatMapPlot(ax_a, linewidth=.1, **group_st['raw'])
        self.__SinglePlot((22, 20), 'corr_a', **group_st['raw'])

        ax_b = fig.add_subplot(gs_1[0, 0])
        self.__AnnotateHeatMap(ax_a, **group_st['sub_0'])
        self.__HeatMapPlot(ax_b, linewidth=.1, **group_st['sub_0'])
        self.__SinglePlot((12, 10), 'corr_b', **group_st['sub_0'])

        ax_c = fig.add_subplot(gs_1[0, 1])
        self.__AnnotateHeatMap(ax_a, **group_st['sub_1'])
        self.__HeatMapPlot(ax_c, linewidth=.1, **group_st['sub_1'])
        self.__SinglePlot((12, 10), 'corr_c', **group_st['sub_1'])

        ax_d = fig.add_subplot(gs_1[1, 0])
        self.__AnnotateHeatMap(ax_a, **group_st['sub_2'])
        self.__HeatMapPlot(ax_d, linewidth=.1, **group_st['sub_2'])
        self.__SinglePlot((8, 10), 'corr_d', **group_st['sub_2'])

        ax_e = fig.add_subplot(gs_1[1, 1])
        self.__AnnotateHeatMap(ax_a, **group_st['sub_3'])
        self.__HeatMapPlot(ax_e, linewidth=.1, **group_st['sub_3'])
        self.__SinglePlot((14, 12), 'corr_e', **group_st['sub_3'])

        fig.tight_layout(pad=2.5)
        fig.savefig(self.__save_p / 'corr_tot.png')


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = FeatCorrMap(mode_n=mode_)
        main_p.Main()
