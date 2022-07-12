'''
Chart 1 Basic patient information statistics
Distribution, Bloodgas, Biochemistr, Medical score
'''

import sys
import pandas as pd
from pathlib import Path
from pylatex.section import Paragraph
from pylatex import Document, Package, LongTable
from pylatex.utils import bold, italic, NoEscape

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, SaveGen
from Classes.ORM.expr import LabExtube, PatientInfo
from Classes.FeatureProcess import FeatureLoader, DatasetGeneration

p_name = 'Chart_1_Baseline'
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
    _, feat_lab = load_p.LabFeatsLoad(PatientInfo, LabExtube)

    feat_map = GetWholeNameMap()
    feat_df = feat_lab[feat_lab.index.isin(feat_map.index)]
    feat_df['para_num'] = feat_df.index.map(feat_map['para_num'])
    feat_df['para_inter'] = feat_df.index.map(feat_map['para_inter'])
    feat_df = feat_df.sort_values(by='para_num')
    feat_df = feat_df.reset_index(drop=True)
    feat_df = feat_df.rename(col_map, axis=1)
    feat_df = feat_df[col_map.values()]

    feat_df.to_csv(save_form / 'dist_table.csv', index=False)
    feat_df.to_latex(save_form / 'dist_table.tex', escape=False, index=False)

    doc = GenLatexPdf(feat_df)
    doc.generate_pdf(str(save_form / 'chart_1'), clean_tex=False)


def GetWholeNameMap() -> dict:
    tot_map = {}
    name_map = ConfigRead(cate='Chart1NameMap', file=json_loc)

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


def GenLatexPdf(df: pd.DataFrame) -> Document:
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''
    Values are given as mean (± sd), median (IQR), or proportion (\%) and 
    compared between the two groups using the Mann-Whitney test or using the 
    students' t-test. '''

    with doc.create(Paragraph('')) as title:
        title.append(bold('Table.1 '))
        title.append(italic('Baseline Characters of sample included'))

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

    with doc.create(Paragraph('')) as tail:
        tail.append(NoEscape(chart_description))

    return doc


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)