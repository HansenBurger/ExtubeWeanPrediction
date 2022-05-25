import sys
import pandas as pd
from pathlib import Path
from itertools import combinations

sys.path.append(str(Path.cwd()))

from Classes.Domain.layer_p import PatientVar
from Classes.Func.KitTools import PathVerify
from Classes.Func.CalculatePart import PerfomAssess
from Classes.MLD.processfunc import DataSetProcess
from Classes.MLD.balancefunc import BalanSMOTE
from Classes.MLD.algorithm import LogisiticReg
from Classes.ORM.expr import LabExtube, PatientInfo


class Basic():
    def __init__(self) -> None:
        pass

    def __TableLoad(self, load_path: Path):
        obj_s = []
        load_path = PathVerify(load_path)
        for file in load_path.iterdir():
            if not file.is_file():
                pass
            elif file.suffix == '.csv':
                obj = PatientVar()
                info_ = file.name.split('_')
                obj.pid = int(info_[0])
                obj.end = int(info_[1])
                obj.icu = str(info_[2])
                obj.data = pd.read_csv(file, index_col='method')
                obj_s.append(obj)

        return obj_s


class FeatureLoader(Basic):
    def __init__(self, local_p: Path):
        super().__init__()
        self.__samples = self._Basic__TableLoad(local_p)

    @property
    def samples(self):
        return self.__samples

    def __GetSampleData(self) -> pd.DataFrame:

        data_ = pd.DataFrame()
        data_['pid'] = [samp.pid for samp in self.__samples]
        data_['end'] = [samp.end for samp in self.__samples]
        data_['icu'] = [samp.icu for samp in self.__samples]
        data_ = data_.sort_values('pid')
        data_ = data_.reset_index(drop=True)

        return data_

    def VarFeatLoad(self, met_s: list = [], ind_s: list = []) -> pd.DataFrame:

        data_var = self.__GetSampleData()
        met_s = met_s if met_s else self.__samples[0].data.index.to_list()
        ind_s = ind_s if ind_s else self.__samples[0].data.columns.to_list()

        for met in met_s:
            for ind in ind_s:
                col_name = met + '-' + ind
                data_var[col_name] = [
                    samp.data.loc[met, ind] for samp in self.__samples
                ]

        return data_var

    def LabFeatLoad(self, src_0: any, src_1: any) -> pd.DataFrame:
        data_lab = self.__GetSampleData()

        join_info = {
            'dest': src_0,
            'on': src_0.pid == src_1.pid,
            'attr': 'pinfo'
        }
        col_order = [src_1.pid]
        col_query = [src_1, src_0.age, src_0.sex, src_0.bmi]
        cond_pid = src_1.pid.in_(data_lab.pid.tolist())

        que_l = src_1.select(*col_query).join(
            **join_info).where(cond_pid).order_by(*col_order)

        data_que = pd.DataFrame(list(que_l.dicts())).drop(['pid'], axis=1)
        data_lab = pd.concat([data_lab, data_que], axis=1)

        return data_lab


class FeatureProcess(Basic):
    def __init__(self,
                 data_: pd.DataFrame,
                 col_label: str,
                 save_path: Path = None):
        super().__init__()
        self.__data = data_
        self.__feat = pd.DataFrame()
        self.__col_l = col_label
        self.__save_p = PathVerify(save_path) if save_path else Path.cwd()

    @property
    def feat(self):
        return self.__feat

    def __SingleLogReg(self, data_: pd.DataFrame, col_l: str, test_s: float):
        X_t, y_t, X_v, y_v = DataSetProcess(data_, col_l).DataSplit(test_s)
        balanced = BalanSMOTE(X_t, y_t)
        train_test = [balanced.X, balanced.y, X_v, y_v]
        model = LogisiticReg(train_test, {'C': 1, 'max_iter': 2000})
        model.Deduce()
        perform_rs = model.Predict()
        auc_v = round(perform_rs['rocauc'], 3)
        auc_diff = round(abs(auc_v - 0.5), 4)

        return auc_v, auc_diff

    def FeatPerformance(self, col_methods: list):

        row_s = []

        for col_met in col_methods:

            df_tmp = self.__data[[self.__col_l, col_met]]
            df_tmp = df_tmp.dropna()

            # Get feature attributes
            n_neg = len(df_tmp[df_tmp[self.__col_l] == 0])
            n_pos = len(df_tmp[df_tmp[self.__col_l] == 1])
            process = PerfomAssess(df_tmp[self.__col_l], df_tmp[col_met])
            auc, _, _, = process.AucAssess()
            p, rs_pos, rs_neg = process.PAssess()
            log_auc, log_diff = self.__SingleLogReg(df_tmp, self.__col_l, 0.3)

            row_value = {
                'met': col_met,
                'P': p,
                'AUC': auc,
                'LogReg': log_auc,
                'LogRegDiff': log_diff,
                'rs_0': rs_neg,
                'size_0': n_neg,
                'rs_1': rs_pos,
                'size_1': n_pos
            }

            row = pd.Series(row_value)
            row_s.append(row)

        self.__feat = pd.DataFrame(row_s)
        pd.DataFrame.to_csv(self.__feat,
                            self.__save_p / 'feature_attr_tot.csv',
                            index=False)

    def DataSelect(self,
                   p_v_max: float = 0.05,
                   diff_min: float = 0.01,
                   feat_lack_max: float = 0.4,
                   recs_lack_max: float = 0.8):

        p_v_filt = self.__feat.P < p_v_max
        diff_filt = self.__feat.LogRegDiff > diff_min
        feats_all = self.__feat[p_v_filt & diff_filt].met.tolist()

        data_ = self.__data[[self.__col_l] + feats_all]
        recs_filt = data_.isnull().sum(axis=1) < data_.shape[1] * recs_lack_max
        feat_filt = data_.isnull().sum(axis=0) < data_.shape[0] * feat_lack_max

        a = self.__data.loc[recs_filt, feat_filt]
        b = 1

        # combine = lambda x, y: [i for i in combinations(x, y)]
        # for i in range(1, len(feats_select) + 1):
        #     mets_l = combine(feats_select, i)


file_rot = r'C:\Main\Data\_\Result\Form\20220524_20_Extube_PSV_Nad'
file_rot_ = r'C:\Main\Data\_\Result\Form\20220524_19_Extube_SumP12_Nad'

get_feat = FeatureLoader(file_rot)
# df_var = get_feat.VarFeatLoad()
df_que = get_feat.LabFeatLoad(PatientInfo, LabExtube)

col_n = df_que.columns.drop(['icu', 'pid', 'end']).to_list()
process_feat = FeatureProcess(df_que, 'end')
feat_p = FeatureProcess(df_que, 'end')
feat_p.FeatPerformance(col_n)
feat_p.DataSelect(0.5, 0.03)
b = feat_p.feat
a = feat_p.feat.loc[feat_p.feat.P < 0.05]

a = 1

# df_xs = df_var.loc[df_var.icu.str.contains('xs')]
# df_qc = df_var.loc[~df_var.icu.str.contains('xs')]

# df_3f = df_var.loc[df_var.icu == 'ICU3F']
# df_4f = df_var.loc[df_var.icu == 'ICU4F']
# df_xs3f = df_var.loc[df_var.icu == 'xsICU3F']
# df_xs4f = df_var.loc[df_var.icu == 'xsICU4F']
# df_eicu = df_var.loc[df_var.icu == 'EICU']

# col_n = df_var.columns.drop(['icu', 'pid', 'end']).to_list()
# feat_p = FeatureProcess(df_eicu, 'end')
# feat_p.FeatPerformance(col_n)

col_n = df_que.columns.drop(['pid'])

b = feat_p.feat
a = feat_p.feat.loc[feat_p.feat.P < 0.05]

b = 1