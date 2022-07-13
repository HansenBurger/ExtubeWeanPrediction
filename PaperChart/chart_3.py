import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from pylatex.section import Paragraph
from pylatex.utils import bold, NoEscape
from pylatex import Document, Package, Figure, SubFigure, Command

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen

p_name = 'Chart_3_ROCAUC'
mode_s = ['Extube_30_SumP12_Nad', 'Extube_60_SumP12_Nad']

pred_models = ['LR', 'RF', 'SVM', 'XGB']
plot_col_map = {
    'auc': 'ROC-AUC',
    'acc': 'Accuracy',
    'sen': 'Sensitivity',
    'spe': 'Specificity'
}
json_loc = Path.cwd() / 'PaperChart' / 'charts.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)
    load_p = Path(ConfigRead('ForwardSearch', mode_, json_loc))

    for p_m in pred_models:
        ave_df = GetAvePerform(p_m, load_p)
        ave_fig = LinesPlot(ave_df, p_m)
        ave_df.to_csv(save_form / (p_m + '.csv'), index=False)
        ave_fig.savefig(save_form / (p_m + '.png'), dpi=300)

    fig_path_s = [save_form / (i + '.png') for i in pred_models]
    doc = GenLatexPdf(fig_path_s)
    doc.generate_pdf(str(save_form / p_name), clean_tex=False)


def GetAvePerform(model_n: str, f_path: Path):
    rows = []
    keys = ['ord', 'var', 'ind', 'sen', 'spe', 'acc', 'auc', 'f_1']
    for folder in f_path.iterdir():
        if not folder.is_dir():
            continue
        else:
            p_ave = pd.read_csv(folder / model_n / 'pred_result.csv').iloc[-1]
            vals_0 = folder.stem.split('-')
            vals_1 = p_ave.loc[['s_sen', 's_spe', 's_acc', 's_auc',
                                's_f_1']].tolist()
            row = pd.DataFrame([dict(zip(keys, vals_0 + vals_1))])
            rows.append(row)
    df = pd.concat(rows, ignore_index=True)
    df['ord'] = df['ord'].astype('int')
    df['ord'] = df['ord'] + 1
    df = df.set_index('ord', drop=True)
    return df


def annot_max(df, max_col, ax=None):
    xmax = df[max_col].idxmax()
    ymax = df[max_col].max()
    text = "Best: $n$ = {:d}, $AUC$ = {:.3f}".format(xmax, ymax)

    bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->",
                      color='black',
                      connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data',
              textcoords='axes fraction',
              arrowprops=arrowprops,
              bbox=bbox_props,
              fontsize=13,
              ha='right',
              va='top')

    ax.annotate(text,
                xy=(xmax, ymax),
                xytext=((xmax / df.shape[0]) + 0.1, ymax - 0.3),
                **kw)


def LinesPlot(df: pd.DataFrame,
              fig_name: str,
              max_col: str = plot_col_map['auc']) -> plt.figure:
    df = df.loc[:, list(plot_col_map.keys())]
    df = df.rename(columns=plot_col_map)
    sns.reset_orig()
    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(18, 6))
    plt.plot([0, df.shape[0]], [0.7, 0.7], 'k--')
    plt.ylim([0.0, 1.1])
    plt.xlim([1, df.shape[0]])
    ax = sns.lineplot(data=df, palette='tab10', linewidth=2.5)
    ax.set_xlabel('Number of variability indicators ($n$)', fontsize=15)
    ax.set_ylabel('Assessed Value', fontsize=15)
    annot_max(df, max_col, ax)
    plt.legend(loc='best')
    title_dict = dict(fontname='Times New Roman', size=18, fontweight='bold')
    plt.title(fig_name, fontdict=title_dict, loc='left')
    plt.tight_layout()
    plt.close()
    return fig


def GenLatexPdf(fig_pathes: list):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''Summary of the results of the forward 
    search, horizontal axis is the number of included variability indicators 
    and vertical axis is the mean value of the five-fold cross-validation. 
    Boxes are the maximum values of the average AUC in the forward search and 
    the corresponding number of indicators. $LR$=logistic regression, 
    $RF$=random forest, $SVM$=support vector machine, $XGB$=XGBooster.'''

    for fig_path in fig_pathes:
        with doc.create(Figure(position='t!')) as row:
            doc.append(Command('centering'))
            row.add_image(str(fig_path), width=NoEscape(r'0.8\linewidth'))

    with doc.create(Paragraph('')) as tail:
        tail.append(bold('Fig.2 '))
        tail.append(NoEscape(chart_description))

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)