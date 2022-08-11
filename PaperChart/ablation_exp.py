import sys
import pandas as pd
from pathlib import Path

from pylatex.utils import bold
from pylatex import Document, Package, LongTable

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen

p_name = 'AblationExp'
mode_s = ['Extube_60_SumP12_Nad']

gp_name_map = {
    'GP1': 'Group 1',
    'GP2': 'Group 2',
    'GP3': 'Group 3',
    'GP4': 'Group 4'
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


class AblationExpSummary(Basic):
    def __init__(self, mode_n: str, model_slt: list) -> None:
        super().__init__()
        self.__models = model_slt
        self.__save_p = s_f_fold / mode_n
        self.__gp_p_s = ConfigRead('AblationExp', mode_n, json_loc)

    def __CollectBestINGP(self, gp_p: Path):
        model_ave = []
        for m in self.__models:
            for f in (gp_p / m).iterdir():
                if not 'Perform' in f.name:
                    continue
                elif not f.suffix == '.csv':
                    continue
                else:
                    df_m = pd.read_csv(f, index_col='mode')
                    d_ave = df_m.loc['ave', :].to_dict()
                    d_ave['model'] = m
                    model_ave.append(d_ave)
        df = pd.DataFrame(model_ave)
        df = df.rename(col_map_s, axis=1)
        df = df[[i for i in col_map_s.values() if i in df.columns]]
        return df

    def Main(self):
        self.__save_p.mkdir(parents=True, exist_ok=True)
        gp_rs = {}
        for k, v in self.__gp_p_s.items():
            save_n = gp_name_map[k]
            gp_rs[save_n] = self.__CollectBestINGP(Path(v))
            gp_rs[save_n].to_csv(self.__save_p / (k + '.csv'), index=False)
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