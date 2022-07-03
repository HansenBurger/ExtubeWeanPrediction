import sys
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import chi2, mutual_info_classif

sys.path.append(str(Path.cwd()))

from Classes.Domain.layer_p import PatientVar
from Classes.Func.KitTools import PathVerify
from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.CalculatePart import PerfomAssess
from Classes.MLD.processfunc import DataSetProcess
from Classes.MLD.balancefunc import BalanSMOTE
from Classes.MLD.algorithm import LogisiticReg
from Classes.ORM.expr import PatientInfo
from Classes.ORM.basic import OutcomeExWean


class Basic():
    def __init__(self) -> None:
        pass

    def __TableLoad(self, load_path: Path):
        obj_s = []
        load_path = PathVerify(load_path)
        for file in load_path.iterdir():
            if not file.is_file() or file.suffix != '.csv':
                pass
            else:
                obj = PatientVar()
                info_ = file.name.split('_')
                obj.pid = int(info_[0])
                obj.end = int(info_[1])
                obj.icu = str(info_[2])
                obj.data = pd.read_csv(file, index_col='method')
                obj_s.append(obj)

        obj_s = sorted(obj_s, key=lambda x: x.pid, reverse=False)
        return obj_s


class FeatureLoader(Basic):
    def __init__(self, local_p: Path):
        super().__init__()
        self.__samples = self._Basic__TableLoad(local_p)
        self.__info_col = ['pid', 'icu', 'end', 'rmk']

    @property
    def samples(self):
        return self.__samples

    @property
    def info_col(self):
        return self.__info_col

    def __GetSampleData(self) -> pd.DataFrame:

        src = PatientInfo
        data_ = pd.DataFrame()
        data_['pid'] = [samp.pid for samp in self.__samples]
        data_['end'] = [samp.end for samp in self.__samples]
        data_['icu'] = [samp.icu for samp in self.__samples]
        c_rmk = src.pid.in_(data_.pid.tolist())
        rmk_ = pd.DataFrame(list(src.select(src.rmk_t).where(c_rmk).dicts()))
        data_ = pd.concat([data_, rmk_], axis=1)
        data_ = data_.rename(columns={'rmk_t': 'rmk'})
        data_ = data_.sort_values('pid')
        data_ = data_.reset_index(drop=True)

        return data_

    def VarFeatLoad(self, met_s: list = [], ind_s: list = []) -> pd.DataFrame:
        '''
        met_s: methods select
        ind_s: indicators select
        '''
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
        '''
        src_0: Patient static data
        src_1: Clinical and physiological data
        '''
        data_lab = self.__GetSampleData()
        data_lab = data_lab.sort_values('pid')

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

        src_2 = OutcomeExWean
        c_mvt = src_2.pid.in_(data_lab.pid.tolist())
        mvt_ = pd.DataFrame(list(
            src_2.select(src_2.mv_t).where(c_mvt).dicts()))

        data_que = pd.DataFrame(list(que_l.dicts())).drop(['pid'], axis=1)
        data_lab = pd.concat([data_lab, data_que, mvt_], axis=1)

        return data_lab

    def WholeFeatLoad(self, data_0: pd.DataFrame,
                      data_1: pd.DataFrame) -> pd.DataFrame:
        data_1 = data_1.drop(['pid', 'end', 'icu'], axis=1)
        data_ = pd.concat([data_0, data_1], axis=1)

        return data_


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

    def FeatPerformance(self, col_methods: list, save_name: str = ''):

        row_s = []
        bin_cols = ['sex', 'gender']
        data_csv = self.__save_p / (save_name + '_data_tot.csv')
        pd.DataFrame.to_csv(self.__data, data_csv, index=False)

        for col_met in col_methods:

            df_tmp = self.__data[[self.__col_l, col_met]]
            df_tmp = df_tmp.dropna()

            # Get feature attributes
            n_neg = len(df_tmp[df_tmp[self.__col_l] == 0])
            n_pos = len(df_tmp[df_tmp[self.__col_l] == 1])

            if n_neg < 2 or n_pos < 2:
                continue

            process = PerfomAssess(df_tmp[self.__col_l], df_tmp[col_met])
            p_cate = 'binary' if col_met in bin_cols else 'continuous'
            p, rs_pos, rs_neg = process.PValueAssess(cate=p_cate)
            auc, _, _, = process.AucAssess()

            row_value = {
                'met': col_met,
                'P': p,
                'AUC': auc,
                'rs_0': rs_neg,
                'size_0': n_neg,
                'rs_1': rs_pos,
                'size_1': n_pos
            }

            row = pd.Series(row_value)
            row_s.append(row)

        self.__feat = pd.DataFrame(row_s)
        if self.__feat.empty:
            pass
        else:
            attr_csv = self.__save_p / (save_name + '_attr_tot.csv')
            self.__feat.to_csv(attr_csv, index=False)

    def DataSelect(self,
                   p_max: float = 0.05,
                   feats_spe: list = [],
                   feat_lack_max: float = 0.4,
                   recs_lack_max: float = 0.2) -> pd.DataFrame:

        if self.__feat.empty:
            return pd.DataFrame()

        # Features select

        if not feats_spe:
            p_v_filt = self.__feat.P < p_max
            filt_cond = p_v_filt
            feats_all = self.__feat[filt_cond].met.tolist()
        else:
            feats_all = feats_spe

        # Data select

        data_ = self.__data[[self.__col_l] + feats_all]
        recs_val = data_.isnull().sum(axis=1) < data_.shape[1] * recs_lack_max
        feat_val = data_.isnull().sum(axis=0) < data_.shape[0] * feat_lack_max
        feats_slt = self.__feat[self.__feat.met.isin(feat_val[feat_val].index)]
        feats_slt = feats_slt.sort_values(by=['P'], ascending=True)
        self.__feat = feats_slt
        feats_slt.to_csv(self.__save_p / 'feature_attr_slt.csv', index=False)

        if feats_slt.empty or len(data_.end.unique()) == 1:
            data_ = pd.DataFrame()
        else:
            data_ = data_.loc[recs_val, feat_val]
            for feat_col in feats_slt.met:
                if data_[feat_col].dtype == 'bool':
                    data_[feat_col] = data_[feat_col].astype(int)

            # feat_violin = self.__save_p / 'feat_violin'
            # feat_violin.mkdir(parents=True, exist_ok=True)

            # plot_p = PlotMain(feat_violin)
            # for feat_col in feats_slt.met:
            #     df_tmp = data_[[self.__col_l, feat_col]]
            #     df_tmp['all'] = ''
            #     plot_p.ViolinPlot(x='all',
            #                       y=feat_col,
            #                       df=df_tmp,
            #                       fig_n=feat_col,
            #                       hue=self.__col_l)

        data_.to_csv(self.__save_p / 'sample_data_slt.csv', index=False)

        return data_

    def DataSelect_ForwardSearch(self,
                                 p_max: float = 0.05,
                                 feat_lack_max: float = 0.4,
                                 recs_lack_max: float = 0.2):
        pass
