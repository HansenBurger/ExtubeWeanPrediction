import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from pylatex import Document, Package, Section, LongTable, Figure, SubFigure, Command
from pylatex.utils import bold, italic, NoEscape
from pylatex.section import Paragraph

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.ORM.expr import LabExtube, PatientInfo
from Classes.FeatureProcess import FeatureLoader, FeatureProcess

p_name = 'Chart_2_Heatmap'
mode_s = [
    'Extube_30_PSV_Nad', 'Extube_30_SumP12_Nad', "Extube_60_PSV_Nad",
    "Extube_60_SumP12_Nad"
]

perform_info = {
    'P': {
        'name': 'p-value',
        'color': 'YlGnBu',
        'range': (0, 0.05)
    },
    'AUC': {
        'name': 'rocauc',
        'color': 'coolwarm',
        'range': ()
    }
}

json_loc = Path.cwd() / 'PaperChart' / 'charts.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)
mets_d = ConfigRead('Chart2NameMap', 'Methods', json_loc)
inds_d = ConfigRead('Chart2NameMap', 'Indicators', json_loc)


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)

    load_p = FeatureLoader(ConfigRead('Source', mode_, json_loc))
    data_ = load_p.VarFeatLoad(mets_d.keys(), inds_d.keys())

    info_col_n = ['pid', 'icu', 'end', 'rmk']
    feat_col_s = data_.columns.drop(info_col_n).tolist()
    feat_var_p = FeatureProcess(data_, 'end', save_form)
    feat_var_p.FeatPerformance(feat_col_s, 'VarFeats')

    feat_df_d = GetVarPerform(feat_var_p.feat.set_index('met', drop=True),
                              list(perform_info.keys()))

    for df_n, df in feat_df_d.items():
        draw_i = perform_info[df_n]
        fig = DrawHeatMap(df, draw_i['color'], draw_i['range'])
        fig.savefig(save_form / (draw_i['name'] + '.png'))
        df.to_csv(save_form / (draw_i['name'] + '.csv'))
        df.to_latex(save_form / (draw_i['name'] + '.tex'))

    doc = GenLatexPdf(save_form / (perform_info['P']['name'] + '.png'),
                      save_form / (perform_info['AUC']['name'] + '.png'))
    doc.generate_pdf(str(save_form / 'chart_2'), clean_tex=False)


def GetVarPerform(df: pd.DataFrame, val_cols: list) -> dict:
    df_d = {}
    for val in val_cols:
        feat_df = pd.DataFrame(columns=['met'] + list(inds_d.keys()))
        feat_df['met'] = mets_d.keys()
        feat_df = feat_df.set_index('met', drop=True)
        for met in feat_df.index:
            for ind in feat_df.columns:
                feat_df.loc[met, ind] = df.loc[('-').join([met, ind]), val]
        feat_df.index = list(mets_d.values())
        feat_df = feat_df.rename(columns=inds_d)
        feat_df = feat_df.astype('float')
        df_d[val] = feat_df
    return df_d


def DrawHeatMap(df: pd.DataFrame, c_map: str, v_range: tuple) -> plt.figure:
    fig_dims = tuple(reversed(df.shape))
    fig_dims = (fig_dims[0] + 1, fig_dims[1])
    df.index = ['$' + i + '$' for i in df.index]
    df.columns = ['$' + i + '$' for i in df.columns]
    fig = plt.figure(figsize=fig_dims, dpi=300)
    res = sns.heatmap(df,
                      annot=True,
                      linewidths=.5,
                      cmap=c_map,
                      annot_kws={'size': 14})
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=12)
    plt.tight_layout()
    plt.close()
    return fig


def GenLatexPdf(fig_path_0: Path, fig_path_1: Path) -> Document:
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

    with doc.create(Paragraph('')) as tail:
        tail.append(bold('Fig.1 '))
        tail.append(NoEscape(chart_description))

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)