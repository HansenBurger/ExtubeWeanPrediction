import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from pylatex.section import Paragraph
from pylatex.utils import bold, italic, NoEscape
from pylatex import Document, Package, Figure, SubFigure, Command, LongTable

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.FeatureProcess import FeatureLoader

p_name = 'RespVariability'
mode_s = ["Extube_60_SumP12_Nad"]

perform_info = {
    'P': {
        'col_n': 'P',
        'color': 'YlGnBu',
        'save_n': 'p-value'
    },
    'AUC': {
        'col_n': 'AUC',
        'color': 'YlOrBr',
        'save_n': 'rocauc'
    },
    'P-AUC': {
        'col_n': ['P', 'AUC'],
        'color': ['YlGnBu', 'YlOrBr'],
        'save_n': 'combine'
    },
    'ind_part': {
        'save_n':
        'resp_ind',
        'mets': ['cv', 'std', 'ave', 'med', 'qua', 'tqua'],
        'inds': [
            'pip', 'rr', 'v_t', 've', 'rsbi', 'wob', 'mp_jl_d', 'mp_jm_d',
            'mp_jl_t', 'mp_jm_t'
        ],
        'p_limit': (0, 0.05),
    },
    'met_part': {
        'save_n': 'resp_met',
        'inds': ['pip', 'rr', 'v_t', 've', 'rsbi'],
        'p_limit': (0, 0.05),
    },
    'new_best': {
        'save_n': 'resp_new',
        'mets': ['pi', 'gi', 'si', 'app', 'samp', 'fuzz', 'ac', 'dc'],
        'inds': ['wob', 'mp_jl_d', 'mp_jm_d', 'mp_jl_t', 'mp_jm_t'],
        'p_limit': (0, 0.05),
    }
}

col_map_s = {
    'para': 'Variability-Respiratory',
    'rs_0': 'Successful N = ',
    'rs_1': 'Failed N = ',
    'P': 'P-value',
    'AUC': 'ROC-AUC'
}

json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)

    main_p = VarResult(mode_, save_form)
    main_p.TotalPerform(**perform_info['P'])
    main_p.TotalPerform(**perform_info['AUC'])
    main_p.TotalPerform(**perform_info['P-AUC'])
    fig_pathes = [
        save_form / i for i in [
            perform_info['P']['save_n'],
            perform_info['AUC']['save_n'],
        ]
    ]
    doc_g_0 = LatexGraph(*fig_pathes)
    doc_g_0.generate_pdf(str(save_form / (p_name + '_graph')), clean_tex=False)

    df_0 = main_p.PartialPerform(**perform_info['ind_part'])
    doc_t_0 = LatexTable(df_0, 3, 'Remarkable respiratory parameters')
    doc_t_0.generate_pdf(str(save_form / (p_name + '_table_0')),
                         clean_tex=False)

    df_1 = main_p.PartialPerform(**perform_info['met_part'])
    doc_t_1 = LatexTable(df_1, 4, 'Significant variability method')
    doc_t_1.generate_pdf(str(save_form / (p_name + '_table_1')),
                         clean_tex=False)

    df_2 = main_p.PartialPerform(**perform_info['new_best'])
    doc_t_2 = LatexTable(df_2, 5, 'Significant rew type')
    doc_t_2.generate_pdf(str(save_form / (p_name + '_table_2')),
                         clean_tex=False)


class Basic():
    def __init__(self) -> None:
        pass

    def GetVarData(self, mode_name: str):
        load_p = FeatureLoader(ConfigRead('Source', mode_name, json_loc))
        data_, feat_ = load_p.VarFeatsLoad()
        feat_['AUC'] = [1 - i if i < 0.5 else i for i in feat_.loc[:, 'AUC']]
        feat_['AUC'] = feat_['AUC'].round(3)
        return data_, feat_


class VarResult(Basic):
    def __init__(self, mode_name: str, save_path: Path) -> None:
        super().__init__()
        _, self.__feat = self.GetVarData(mode_name)
        self.__save_p = save_path
        self.__name_map = {
            'mets': ConfigRead('RespVar', 'Methods', json_loc),
            'inds': ConfigRead('RespVar', 'Indicators', json_loc)
        }

    def __HeatmapPlot(self, df: pd.DataFrame, color: str, ax: any):
        df = df.copy()
        df.index = ['$' + i + '$' for i in df.index]
        df.columns = ['$' + i + '$' for i in df.columns]
        sns.heatmap(df,
                    annot=True,
                    linewidths=.5,
                    cmap=color,
                    annot_kws={'size': 14},
                    ax=ax)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=12)
        return ax

    def __GetPerformance(self, col_n: str, save_n: str = ''):
        var_cols = ['met'] + list(self.__name_map['inds'].keys())
        df_var = pd.DataFrame(columns=var_cols)
        df_var['met'] = self.__name_map['mets'].keys()
        df_var = df_var.set_index('met', drop=True)

        for met in df_var.index:
            for ind in df_var.columns:
                df_var.loc[met, ind] = self.__feat.loc[('-').join([met, ind]),
                                                       col_n]

        df_var.index = list(self.__name_map['mets'].values())
        df_var = df_var.rename(columns=self.__name_map['inds'])
        df_var = df_var.astype('float')

        if not save_n:
            pass
        else:
            df_var.to_csv(self.__save_p / (save_n + '.csv'))
            df_var.to_latex(self.__save_p / (save_n + '.tex'), escape=False)

        return df_var

    def TotalPerform(self, col_n: list, color: list, save_n: str):
        col_n = [col_n] if type(col_n) != list else col_n
        color = [color] if type(color) != list else color

        df_s = []
        for col in col_n:
            df = self.__GetPerformance(
                col, save_n=None if len(col_n) > 1 else save_n)
            df_s.append(df)

        fig_dims = tuple(reversed(df_s[0].shape))
        fig_dims = (fig_dims[0] * len(df_s) + 1, fig_dims[1])
        fig, ax_s = plt.subplots(1, len(df_s), figsize=fig_dims, dpi=300)
        for i in range(len(col_n)):
            try:
                ax_s[i] = self.__HeatmapPlot(df_s[i], color[i], ax_s[i])
            except:
                ax_s = self.__HeatmapPlot(df_s[i], color[i], ax_s)
        plt.tight_layout()
        fig.savefig(self.__save_p / (save_n + '.png'))

    def PartialPerform(
        self,
        save_n: str,
        mets: list = [],
        inds: list = [],
        respvar_s: list = [],
        p_limit: tuple = (0, 1.1),
        auc_limit: tuple = (0, 1.1),
        order_by: str = 'P',
        show_stop: int = -1,
    ):

        if respvar_s:
            df = self.__feat.loc[respvar_s, :]
        else:
            mets = list(self.__name_map['mets'].keys()) if not mets else mets
            inds = list(self.__name_map['inds'].keys()) if not inds else inds
            index_raw = []
            index_inter = []
            for i in mets:
                for j in inds:
                    met = self.__name_map['mets'][i]
                    ind = self.__name_map['inds'][j]
                    index_raw.append('-'.join([i, j]))
                    index_inter.append('-'.join([met, ind]))
            df = self.__feat.loc[index_raw, :]
            df['para'] = index_inter

        filt_p = (df.P >= p_limit[0]) & (df.P <= p_limit[1])
        filt_auc = (df.AUC >= auc_limit[0]) & (df.AUC <= auc_limit[1])
        df = df.loc[filt_p & filt_auc]
        df = df.sort_values(by=order_by, ascending=True, axis=0)

        if df.empty:
            pass
        else:
            col_map_s_ = col_map_s.copy()
            col_map_s_['rs_0'] = col_map_s_['rs_0'] + str(df['size_0'][0])
            col_map_s_['rs_1'] = col_map_s_['rs_1'] + str(df['size_1'][0])
            df = df.rename(col_map_s_, axis=1)
            df = df[[i for i in col_map_s_.values() if i in df.columns]]
            df = df.loc[0:show_stop, :] if show_stop > 0 else df.loc[:, :]

            df.to_csv(self.__save_p / (save_n + '.csv'), index=False)
            df.to_latex(self.__save_p / (save_n + '.tex'),
                        escape=False,
                        index=False)

        return df


def LatexGraph(fig_path_0: Path, fig_path_1: Path) -> Document:
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''Performance statistics of variability metrics, 
    including p-value and roc-auc.Variability statistics are, $AVE$=mean,
    $STD$=standard deviation, $CV$=coefficient of variation, $QUA$=lower quartile,
    $TQUA$=upper quartile, $SD_1$=dispersion of semi-long axis,
    $SD_2$=dispersion of semi-short axis, $PI$, $GI$, $SI$ are three asymmetry 
    analysis methods, $AppEn$=approximate entropy, $SampEn$=sample entropy,
    $FuzzEn$=fuzzy entropy, $PRSA(AC)$=Phase-rectified signal averaging (Increase),
    $PRSA(DC)$=Phase-rectified signal averaging (Decrease).The incorporated respiratory
    indices are, $RR$=Respiratory rate, $V_t$=Tidal volume, $MV$=Minute ventilation,
    $WOB$=Work of breathing, $MP_d$=Mechanical work (Dynamic),
    $MP_t$=Mechanical work (Total)'''

    with doc.create(Figure(position='h!')) as image_map:
        doc.append(Command('centering'))
        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.55\linewidth'))) as left_row:
            left_row.add_image(str(fig_path_0), width=NoEscape(r'1\linewidth'))
            left_row.add_caption('P-value HeatMap')

        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.55\linewidth'))) as right_row:
            right_row.add_image(str(fig_path_1),
                                width=NoEscape(r'1\linewidth'))
            right_row.add_caption('AUC HeatMap')

    # with doc.create(Paragraph('')) as tail:
    #     tail.append(bold('Fig.3 '))
    #     tail.append(NoEscape(chart_description))

    return doc


def LatexTable(df: pd.DataFrame, table_ind: int, table_name: str) -> Document:
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    # with doc.create(Paragraph('')) as title:
    #     title.append(bold('Table.{0} '.format(table_ind)))
    #     title.append(italic(table_name))

    with doc.create(LongTable('l|ll|r|r', row_height=1.5)) as data_table:
        data_table.add_hline()
        data_table.add_row(df.columns.to_list(), mapper=bold)
        data_table.add_hline()
        for i in df.index:
            row = df.loc[i, :]
            row[col_map_s['para']] = '$' + row[col_map_s['para']] + '$'
            row = row.values.flatten().tolist()
            data_table.add_row(row, escape=False)
        data_table.add_hline()

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)