import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import SaveGen, ConfigRead

fold_p = Path(
    r'C:\Main\Data\_\Result\Graph\20220701_22_VarAnalysis_60min\Extube_SumP12_Nad'
)
p_name = 'Corr_MapPorcess'
save_p = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)
group_range = {
    '1': {
        'row': slice(0, 54),
        'col': slice(0, 54),
        'cmp': 'coolwarm',
        'shape': (15, 13)
    },
    '1-1': {
        'row': slice(0, 18),
        'col': slice(0, 18),
        'cmp': 'coolwarm',
        'shape': (8, 6)
    },
    '1-2': {
        'row': slice(18, 36),
        'col': slice(18, 36),
        'cmp': 'rocket',
        'shape': (8, 6)
    },
    '2': {
        'row': slice(18, 36),
        'col': slice(81, 144),
        'cmp': 'coolwarm',
        'shape': (18, 6)
    },
    '2-1': {
        'row': slice(18, 36),
        'col': slice(81, 99),
        'cmp': 'rocket',
        'shape': (8, 6)
    },
    '2-2': {
        'row': slice(18, 36),
        'col': slice(117, 144),
        'cmp': 'coolwarm',
        'shape': (10, 6)
    },
    '3': {
        'row': slice(81, 144),
        'col': slice(81, 144),
        'cmp': 'coolwarm',
        'shape': (15, 13)
    }
}


def PretableGen(table_n: str):
    df = pd.read_csv(fold_p / (table_n + '.csv'))
    df['corr'] = df.columns
    df = df.set_index('corr', drop=True)
    return df


def ImpFiltTable(df: pd.DataFrame):
    imp_filt = df.abs() > 0.6
    df_imp = df[imp_filt]
    df_imp = df_imp.fillna(0)
    return df_imp


def ImpSortSave(df: pd.DataFrame, save_p: Path):
    df_corr_imp = pd.DataFrame({})
    i_met, i_ind, j_met, j_ind, corr_v = [], [], [], [], []
    for i in df.index:
        for j in df.columns:
            if i == j:
                continue
            elif df.loc[i, j] == 0:
                continue
            else:
                i_met.append(i.split('-')[0])
                i_ind.append(i.split('-')[1])
                j_met.append(j.split('-')[0])
                j_ind.append(j.split('-')[1])
                corr_v.append(df.loc[i, j])
    df_corr_imp['i_met'] = i_met
    df_corr_imp['i_ind'] = i_ind
    df_corr_imp['j_met'] = j_met
    df_corr_imp['j_ind'] = j_ind
    df_corr_imp['corr_v'] = corr_v
    df_corr_imp_ac = df_corr_imp.sort_values(by=['corr_v'], ascending=True)
    df_corr_imp_dc = df_corr_imp.sort_values(by=['corr_v'], ascending=False)
    df_corr_imp_ac.to_csv(save_p / 'ac.csv', index=False)
    df_corr_imp_dc.to_csv(save_p / 'dc.csv', index=False)


def HeatMapPlot(df: pd.DataFrame, fig_dim: tuple, cmp: str, save_p: Path):
    plt.subplots(figsize=fig_dim)
    sns.reset_orig()
    sns.heatmap(df, annot=False, linewidths=.3, cmap=cmp)
    plt.tight_layout()
    plt.savefig(save_p / 'Heatmap.png')
    plt.close()


def main():
    spearman = PretableGen('CorrSpearman')
    spearman_imp = ImpFiltTable(spearman)
    spearman_save = save_p / 'Total'
    spearman_save.mkdir(parents=True, exist_ok=True)
    HeatMapPlot(spearman_imp, (22, 20), 'coolwarm', spearman_save)
    for k, v in group_range.items():
        row = spearman_imp.columns[v['row']]
        col = spearman_imp.columns[v['col']]
        k_imp = spearman_imp.loc[row, col]
        k_save = save_p / ('G' + k)
        k_save.mkdir(parents=True, exist_ok=True)
        HeatMapPlot(k_imp, v['shape'], v['cmp'], k_save)
        ImpSortSave(k_imp, k_save)


if __name__ == '__main__':
    main()