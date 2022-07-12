import sys
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen

p_name = 'Chart_3_ROCAUC'
mode_s = ['Extube_30_SumP12_Nad', 'Extube_60_SumP12_Nad']

pred_models = ['LR', 'RF', 'SVM', 'XGB']
json_loc = Path.cwd() / 'PaperChart' / 'charts.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main(mode_: str):
    load_p = Path(ConfigRead('ForwardSearch', mode_, json_loc))
    for p_m in pred_models:
        GetAvePerform(p_m, load_p)


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
    return df


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)