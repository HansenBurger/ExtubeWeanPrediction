'''
Chart 1 Basic patient information statistics
Distribution, Bloodgas, Biochemistr, Medical score
'''

import sys
import pandas as pd
from pathlib import Path
from pylatex import Document, Package, Section, NoEscape, LongTabu, LongTable
from pylatex.utils import bold, NoEscape
from matplotlib import pyplot as plt

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.ORM.expr import LabExtube, PatientInfo
from Classes.FeatureProcess import FeatureLoader, FeatureProcess

p_name = 'Chart_1_PatientStatic'
mode_s = [
    'Extube_30_PSV_Nad', 'Extube_30_SumP12_Nad', "Extube_60_PSV_Nad",
    "Extube_60_SumP12_Nad"
]
map_cates = [
    'BG-AcidAlkaline', 'BG-Electrolytes', 'BG-Oxygenation', 'BG-Other',
    'Biochem', 'Unkown', 'MedicalScore'
]
col_map = {
    'para_inter': 'Variable',
    'rs_0': 'Successful extubation',
    'size_0': 'Successful (n)',
    'rs_1': 'Failed extubation',
    'size_1': 'Failed (n)',
    'P': 'P-value',
}

json_loc = Path.cwd() / 'PaperChart' / 'charts.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)

    load_p = FeatureLoader(ConfigRead('Source', mode_, json_loc))
    data_ = load_p.LabFeatLoad(PatientInfo, LabExtube)

    info_col_n = ['pid', 'icu', 'end', 'rmk']
    feat_col_s = data_.columns.drop(info_col_n).tolist()
    feat_lab_p = FeatureProcess(data_, 'end', save_form)
    feat_lab_p.FeatPerformance(feat_col_s, 'LabFeats')

    feat_map = GetWholeNameMap()
    feat_df = feat_lab_p.feat[feat_lab_p.feat.met.isin(feat_map.index)]
    feat_df['para_num'] = feat_df['met'].map(feat_map['para_num'])
    feat_df['para_inter'] = feat_df['met'].map(feat_map['para_inter'])
    feat_df = feat_df.sort_values(by='para_num')
    feat_df = feat_df.reset_index(drop=True)
    feat_df = feat_df.rename(col_map, axis=1)
    feat_df = feat_df[col_map.values()]

    pd.DataFrame.to_csv(feat_df, save_form / 'dist_table.csv', index=False)
    pd.DataFrame.to_latex(feat_df,
                          save_form / 'dist_table.tex',
                          escape=False,
                          index=False)

    doc = GenLatexPdf(feat_df)
    doc.generate_pdf(str(save_form / 'dist_full'), clean_tex=False)


def GetWholeNameMap():
    tot_map = {}
    name_map = ConfigRead(cate='LabFeatNameMap', file=json_loc)

    for map_ in name_map.keys():
        for k, v in name_map.get(map_).items():
            tot_map.setdefault(k, []).append(v)

    tot_mapping = pd.DataFrame({
        'para': tot_map.keys(),
        'para_inter': [i[0] for i in tot_map.values()]
    })
    tot_mapping['para_num'] = tot_mapping.index
    tot_mapping['para_inter'] = tot_mapping['para_inter'].astype('str')
    tot_mapping = tot_mapping.set_index('para', drop=True)

    return tot_mapping


def GenLatexPdf(df: pd.DataFrame):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    with doc.create(LongTable('llrlrr', row_height=1.5)) as data_table:
        data_table.add_hline()
        data_table.add_row(df.columns.to_list(), mapper=bold)
        data_table.add_hline()
        for i in df.index:
            row = df.loc[i, :]
            row.Variable = '$' + row.Variable + '$'
            row = row.values.flatten().tolist()
            data_table.add_row(row, escape=False)
        data_table.add_hline()

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)