import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from functools import reduce
from matplotlib import pyplot as plt

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.FeatureProcess import FeatureLoader

p_name = 'DifferScale'
p_mode = 'Extube_60_SumP12_Nad'
json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)
param_slt = ['mp_jl_d', 'mp_jm_d', 'mp_jb_d', 'pip', 'rr', 'v_t', 've', 'rsbi']

main_folders = [
    '20220904_16_VarAnalysis_60m_2s', '20230118_00_VarAnalysis_60m_5s',
    '20230315_10_VarAnalysis_60m_8s', '20230315_11_VarAnalysis_60m_16s',
    '20230315_12_VarAnalysis_60m_32s', '20230315_13_VarAnalysis_60m_64s',
    '20230315_21_VarAnalysis_60m_128s'
]
sub_folder = 'Chart\Extube_SumP12_Nad'
main_path = Path(r'C:\Main\Data\_\Result\Mix')
pathes = [main_path / f / sub_folder for f in main_folders]


def get_bv(df: pd.DataFrame):
    inds = df.columns.tolist()
    mets = df.index.tolist()
    cols, vals = [], []
    for m in mets:
        for i in inds:
            cols.append('-'.join([m, i]))
            vals.append(df.loc[m, i])
    row = pd.Series(dict(zip(cols, vals)))
    return row


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)
    main_p = VarScaleResult(mode_, save_form)
    main_p.LinePlot()


class Basic():
    def __init__(self) -> None:
        pass

    def GetVarData(self, mode_name: str, param_s: list = param_slt):
        data_ = pd.read_csv(
            Path(ConfigRead('VarRange', mode_name, json_loc)) / 'main.csv')
        feat_series = [data_[i][0:30] for i in data_.columns if 'feat' in i]
        feat_ = reduce(np.intersect1d, (feat_series))
        feat_ = [i for i in feat_ if i.split('-')[1] in param_s]
        return data_, feat_


class VarScaleResult(Basic):
    def __init__(self, mode_name: str, save_path: Path) -> None:
        super().__init__()
        self.__save_p = save_path
        self.__data, self.__feat = self.get_table()

    def __ResetNameMap(self, list_raw: list):
        # list_raw = self.__ExcludeWob(list_raw)
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

            rename_st = '$' + met_inter + '-' + ind_inter + '$'
            # rename_st = '$' + met_inter + '-' + ind_inter + unit_st + '$'
            list_rename.append(rename_st)

        new_name_map = dict(zip(list_raw, list_rename))
        return new_name_map

    def get_table(self, pathes: list = pathes, param_s: list = param_slt):
        df_s = []
        for path in pathes:
            row_p = get_bv(
                pd.read_csv(path / 'P_HeatMap.csv', index_col='method'))
            row_auc = get_bv(
                pd.read_csv(path / 'AUC_HeatMap.csv', index_col='method'))
            type_n = path.parts[6].split('_')[-1]

            with open(self.__save_p / 'info.txt', 'a') as f:
                info_l = [
                    type_n, row_p[row_p < 0.01].shape[0],
                    row_p[row_p < 0.05].shape[0]
                ]
                f.write('Scale: {0}\tP < 0.01: {1}\tP < 0.05: {2}\n'.format(
                    *info_l))

            col_n = [type_n + '_' + i for i in ['feat', 'p', 'auc']]
            val_s = [
                row_p.index.to_list(),
                row_p.values.tolist(),
                row_auc.values.tolist()
            ]
            df_p = pd.DataFrame(dict(zip(col_n, val_s)))
            df_p = df_p.sort_values(by=col_n[1])
            df_p = df_p.reset_index(drop=True)
            df_p.to_csv(self.__save_p / (type_n + '_scale.csv'), index=False)
            df_s.append(df_p)

        data = pd.concat(df_s, axis=1)
        data.to_csv(self.__save_p / 'main.csv', index=False)

        feat_series = [data[i][0:30] for i in data.columns if 'feat' in i]
        feat = reduce(np.intersect1d, (feat_series))
        feat = [i for i in feat if i.split('-')[1] in param_s]
        return data, feat

    def LinePlot(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        scale_s = [2, 5, 8, 16, 32, 64, 128]
        feat_re = self.__ResetNameMap(self.__feat)
        colors = ['dodgerblue', 'limegreen', 'darkorange', 'firebrick']
        for i in range(len(feat_re)):
            values = []
            for s in scale_s:
                df_p = self.__data[[
                    i for i in self.__data.columns if str(s) in i
                ]]
                df_p = df_p.set_index(df_p.columns[0], drop=True)
                v = df_p.loc[self.__feat[i], str(s) + 's_auc']
                values.append(v)
            sns.lineplot()
            ax.plot(scale_s,
                    values,
                    label=feat_re[self.__feat[i]],
                    linewidth=3,
                    c=colors[i])
            ax.plot(scale_s, values, 'o', c=colors[i], markersize=10)
        label_st = dict(fontname='Times New Roman', style='normal', size=20)
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.set_xlabel('Scale (s)', fontdict=label_st)
        ax.set_ylabel('AUC', fontdict=label_st)
        fig.tight_layout()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(prop={'size': 14})
        ax.set_xscale('log', base=2)
        fig.savefig(self.__save_p / 'main.png', dpi=300)
        plt.close()


if __name__ == "__main__":
    main(p_mode)