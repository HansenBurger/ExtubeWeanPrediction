import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt

from pylatex.utils import bold
from pylatex import Document, Package, LongTable

sys.path.append(str(Path.cwd()))

from Classes.FeatureProcess import FeatureLoader
from Classes.Func.KitTools import ConfigRead, SaveGen

p_name = 'AblationExp'
mode_s = ['Extube_60_SumP12_Nad']

gp_name_map = {
    'GP1': 'Group A',
    'GP2': 'Group A+B',
    'GP3': 'Group A+C',
    'GP4': 'Group A+B+C+D'
}

pred_models = ['LR', 'RF', 'SVM', 'XGB']
col_map_s = {
    'model': 'model',
    's_auc': 'AUC',
    's_acc': 'ACC',
    's_sen': 'SEN',
    's_spe': 'SPE'
}

json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


class Basic():
    def __init__(self) -> None:
        pass

    def __ResultsGetByName(self, path_: Path, models: list,
                           file_name_key: str):
        file_d = {}
        for m in models:
            for f in (path_ / m).iterdir():
                if not file_name_key in f.name:
                    continue
                elif not f.suffix == '.csv':
                    continue
                else:
                    file_d[m] = f
        return file_d


class AblationExpSummary(Basic):
    def __init__(self, mode_n: str, model_slt: list) -> None:
        super().__init__()
        self.__models = model_slt
        self.__save_p = s_f_fold / mode_n
        self.__feat_loader = FeatureLoader(
            Path(ConfigRead('Source', mode_n, json_loc)))
        self.__gp_p_s = ConfigRead('AblationExp', mode_n, json_loc)

    def __ResetNameMap(self, list_raw: list):
        list_raw = self.__ExcludeWob(list_raw)
        total_name_map = {
            'mets': ConfigRead('RespVar', 'Methods', json_loc),
            'inds': ConfigRead('RespVar', 'Indicators', json_loc)
        }
        list_rename = []
        for i in list_raw:
            [met_n, ind_n] = i.split('-')
            met_inter = total_name_map['mets'][met_n]['Inter']
            met_unit = total_name_map['mets'][met_n]['Unit']
            ind_inter = total_name_map['inds'][ind_n]['Inter']
            ind_unit = total_name_map['inds'][ind_n]['Unit']

            if not met_unit:
                unit_st = ''
            elif met_unit == '-':
                unit_st = '(' + ind_unit + ')'
            elif met_unit == '%':
                unit_st = '(' + '\%' + ')'
            else:
                unit_st = '(' + met_unit + ')'

            rename_st = '$' + met_inter + '-' + ind_inter + unit_st + '$'
            list_rename.append(rename_st)

        new_name_map = dict(zip(list_raw, list_rename))
        return new_name_map

    def __ExcludeWob(self, list_):
        list_ = [
            i.split('-')[0] + '-' +
            'mp_jb_d' if i.split('-')[1] == 'wob' else i for i in list_
        ]
        return list_

    def __ImpFeatCorr(self, df):
        imp_cols = df.index.tolist()
        imp_cols = self.__ExcludeWob(imp_cols)
        df_data, _ = self.__feat_loader.VarFeatsLoad(spec_=imp_cols)
        df_data = self.__feat_loader.DropInfoCol(df_data)
        df_corr = df_data.corr(method='spearman')
        df_corr = df_corr[imp_cols]
        df_corr = df_corr.loc[imp_cols, :]
        return df_corr

    def __CollectBestInGP(self, gp_p: Path):
        model_ave = []
        file_d = self._Basic__ResultsGetByName(gp_p, self.__models, 'Perform')
        for m, f in file_d.items():
            df_m = pd.read_csv(f, index_col='mode')
            d_ave = df_m.loc['ave', :].to_dict()
            d_ave = dict(
                zip(list(d_ave.keys()), [
                    str(v) + '±' +
                    str(round(np.std(df_m[k].to_list()[:-1]), 3))
                    for k, v in d_ave.items()
                ]))
            d_ave['model'] = m
            model_ave.append(d_ave)

        df = pd.DataFrame(model_ave)
        df = df.rename(col_map_s, axis=1)
        df = df[[i for i in col_map_s.values() if i in df.columns]]
        return df

    def __BarPlot(self, ax: any, **kwargs) -> None:
        sns.barplot(ax=ax, **kwargs)
        label_st = dict(family='Arial', style='normal', size=20)
        ax.set_ylabel('Breathing Variability Indices', fontdict=label_st)
        ax.set_xticks(ax.get_xticks(), fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        # ax.set_xlabel('(a) Features\' Importance', fontdict=label_st)

    def __HeatMapPlot(self, ax: any, **kwargs) -> None:
        color_st = [
            'royalblue', 'cornflowerblue', 'lightsteelblue', 'darksalmon',
            'coral', 'orangered'
        ]
        cmap = (mpl.colors.ListedColormap(color_st))
        bounds = [-1.0, -0.8, -0.6, 0.0, 0.6, 0.8, 1.0]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sns.heatmap(cmap=cmap, norm=norm, ax=ax, **kwargs)
        label_st = dict(family='Arial', style='normal', size=20)
        # ax.set_xlabel('(b) Features\' Correlation', fontdict=label_st)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=10)

    def __DrawImpAndCorr(self, gp_p: Path):
        fig_d = {}
        file_d = self._Basic__ResultsGetByName(gp_p, self.__models, 'Import')
        for m, f in file_d.items():
            if m != 'XGB':
                continue
            df_m = pd.read_csv(f, index_col=0)
            df_m = df_m.sort_values(by=['ave'], ascending=False)
            df_corr = self.__ImpFeatCorr(df_m)
            name_map = self.__ResetNameMap(df_m.index.to_list())
            df_m.index = list(name_map.values())
            df_corr.index = list(name_map.values())
            df_corr.rename(columns=name_map, inplace=True)
            df_corr.columns = df_corr.index.to_list()

            cube_size = 0.8 if df_m.shape[0] > 10 else 1.2
            heat_dims = [df_m.shape[0] * cube_size] * 2
            dim_st = (8 + heat_dims[0], heat_dims[1])
            fig, (ax_0, ax_1) = plt.subplots(
                1,
                2,
                figsize=dim_st,
                gridspec_kw={'width_ratios': [8 / heat_dims[0], 1.2]})

            self.__BarPlot(ax_0, y=df_m.index.to_list(), x=df_m.ave.to_list())
            self.__HeatMapPlot(ax_1, linewidth=.5, data=df_corr)
            fig.tight_layout()
            fig_d[m] = fig

        return fig_d

    def Main(self):
        self.__save_p.mkdir(parents=True, exist_ok=True)
        gp_rs = {}
        for k, v in self.__gp_p_s.items():
            if k != 'GP4':
                continue
            save_n = gp_name_map[k]
            gp_rs[save_n] = self.__CollectBestInGP(Path(v))
            gp_rs[save_n].to_csv(self.__save_p / (k + '.csv'), index=False)
            fig_d = self.__DrawImpAndCorr(Path(v))
            for m, f in fig_d.items():
                f.savefig(self.__save_p / (k + '_' + m + '.png'), dpi=300)
        doc = LatexTable(gp_rs)
        doc.generate_pdf(str(self.__save_p / (p_name + '_flow')),
                         clean_tex=False)


def LatexTable(table_d: dict):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    with doc.create(LongTable('l|rrrr', row_height=1.5)) as data_table:
        data_table.add_hline()
        cols = [''] + list(table_d.values())[0].columns.to_list()[1:]
        data_table.add_row(cols, mapper=bold)
        data_table.add_hline()
        for k, v in table_d.items():
            data_table.add_row([k] + [''] * (v.shape[1] - 1))
            for i in v.index:
                row = v.loc[i, :]
                indent = '\hspace{5mm}'
                row.model = '$' + indent + row.model + '$'
                row = row.values.flatten().tolist()
                data_table.add_row(row, escape=False)
            data_table.add_hline()
        data_table.add_hline()

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main_p = AblationExpSummary(mode_, pred_models)
        main_p.Main()