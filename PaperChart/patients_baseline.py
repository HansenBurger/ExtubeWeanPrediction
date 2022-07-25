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

p_name = 'PatientsBaseline'
mode_s = ["Extube_60_SumP12_Nad"]

table_info = {
    'statistic': {
        'cate': 'Baseline',
        'item': ['Statistics'],
        'name': '0_BL_table'
    },
    'remark': {
        'cate': 'Baseline',
        'item': ['Remarks'],
        'name': '0_RMK_table'
    },
    'BGBC': {
        'cate':
        'BGBC',
        'item': [
            'BG-AcidAlkaline', 'BG-Electrolytes', 'BG-Oxygenation', 'BG-Other',
            'Biochem', 'Unkown'
        ],
        'name':
        '1_BGBC_table'
    }
}

col_map_s = {
    'para_inter': 'Variable',
    'rs_0': 'Successful extubation',
    'size_0': 'Successful (n)',
    'rs_1': 'Failed extubation',
    'size_1': 'Failed (n)',
    'P': 'P-value',
}

json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main(mode_: str):
    save_form = s_f_fold / mode_
    save_form.mkdir(parents=True, exist_ok=True)

    main_p = DistTable(mode_, save_form)
    doc_0 = LatexTable_0(*main_p.BaseLineTable())
    doc_0.generate_pdf(str(save_form / (p_name + '_0')), clean_tex=False)
    doc_1 = LatexTable_1(main_p.BGBiochemTable())
    doc_1.generate_pdf(str(save_form / (p_name + '_1')), clean_tex=False)


class Basic():
    def __init__(self) -> None:
        pass

    def GetDistData(self, mode_name: str):
        load_p = FeatureLoader(ConfigRead('Source', mode_name, json_loc))
        data_, feat_ = load_p.LabFeatsLoad(PatientInfo, LabExtube)
        return data_, feat_

    def GetNameMap(self, cate_name: str, item_names: list = []):
        tot_map = {}
        name_map = ConfigRead(cate=cate_name, file=json_loc)

        map_s = name_map.keys() if not item_names else item_names
        for map_ in map_s:
            for k, v in name_map.get(map_).items():
                tot_map.setdefault(k, []).append(v)

        tot_mapping = pd.DataFrame({
            'para':
            tot_map.keys(),
            'para_inter': [i[0] for i in tot_map.values()]
        })
        tot_mapping['para_num'] = tot_mapping.index
        tot_mapping['para_inter'] = tot_mapping['para_inter'].astype('str')
        tot_mapping = tot_mapping.set_index('para', drop=True)

        return tot_mapping


class DistTable(Basic):
    def __init__(self, mode_name: str, save_path: Path) -> None:
        super().__init__()
        self.__data, self.__feat = self.GetDistData(mode_name)
        self.__save_p = save_path

    def __RenameByMap(self, source_: pd.DataFrame, **kwargs):
        map_ = self.GetNameMap(**kwargs)
        df_ = source_[source_.index.isin(map_.index)]
        df_['para_num'] = df_.index.map(map_['para_num'])
        df_['para_inter'] = df_.index.map(map_['para_inter'])
        df_ = df_.sort_values(by='para_num')
        df_ = df_.reset_index(drop=True)
        df_ = df_.rename(col_map_s, axis=1)
        df_ = df_[[i for i in col_map_s.values() if i in df_.columns]]
        return df_

    def __SaveToLocal(self, df: pd.DataFrame, df_i: dict) -> pd.DataFrame:
        df_re = self.__RenameByMap(source_=df,
                                   cate_name=df_i['cate'],
                                   item_names=df_i['item'])
        df_re.to_csv(self.__save_p / (df_i['name'] + '.csv'), index=False)
        df_re.to_latex(self.__save_p / (df_i['name'] + '.tex'),
                       escape=False,
                       index=False)
        return df_re

    def BaseLineTable(self):
        rmk_d_s = []
        for rmk in self.__data.rmk.unique():
            c_succ = self.__data.end == 0
            c_fail = self.__data.end == 1
            c_rmk = self.__data.rmk == rmk
            rmk_d = {}
            rmk_d['rmk'] = rmk
            rmk_d['rs_0'] = '-'
            rmk_d['size_0'] = self.__data[c_rmk & c_succ].shape[0]
            rmk_d['rs_1'] = '-'
            rmk_d['size_1'] = self.__data[c_rmk & c_fail].shape[0]
            rmk_d['P'] = '-'
            rmk_d_s.append(rmk_d)
        df_rmk = pd.DataFrame(rmk_d_s)
        df_rmk = df_rmk.set_index('rmk', drop=True)

        df_0 = self.__SaveToLocal(self.__feat, table_info['statistic'])
        df_1 = self.__SaveToLocal(df_rmk, table_info['remark'])

        return df_0, df_1

    def BGBiochemTable(self):
        df_0 = self.__SaveToLocal(self.__feat, table_info['BGBC'])
        return df_0


def LatexTable_0(df_0: pd.DataFrame, df_1: pd.DataFrame) -> Document:
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    with doc.create(Paragraph('')) as title:
        title.append(bold('Table.1 '))
        title.append(italic('Demographics of the datasets'))

    with doc.create(LongTable('llrlrr', row_height=1.5)) as data_table:
        data_table.add_hline()
        data_table.add_row(df_0.columns.to_list(), mapper=bold)
        data_table.add_hline()
        for i in df_0.index:
            row = df_0.loc[i, :]
            row.Variable = '$' + row.Variable + '$'
            row = row.values.flatten().tolist()
            data_table.add_row(row, escape=False)
        data_table.add_row(['Reason for MV (n)'] + [''] * (len(row) - 1))
        for i in df_1.index:
            row = df_1.loc[i, :]
            indent = '\hspace{5mm}'
            row.Variable = '$' + indent + row.Variable + '$'
            row = row.values.flatten().tolist()
            data_table.add_row(row, escape=False)
        data_table.add_hline()

    return doc


def LatexTable_1(df: pd.DataFrame) -> Document:
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''
    Values are given as mean (± sd), median (IQR), or proportion (\%) and
    compared between the two groups using the Mann-Whitney test or using the
    students' t-test. '''

    with doc.create(Paragraph('')) as title:
        title.append(bold('Table.2 '))
        title.append(italic('Baseline characteristics of the patients enroll'))

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