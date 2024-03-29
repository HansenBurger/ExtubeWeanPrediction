import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.ORM.expr import PatientInfo
from Classes.ORM.basic import OutcomeExWean
from Classes.Domain.layer_p import PatientVar
from Classes.Func.KitTools import PathVerify
from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.CalculatePart import PerfomAssess


class Basic():
    def __init__(self) -> None:
        self.__label_col = 'end'
        self.__info_cols = ['pid', 'icu', 'end', 'rmk']

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

    def __GetSampleData(self, samples: any) -> pd.DataFrame:

        src = PatientInfo
        data_ = pd.DataFrame()
        data_['pid'] = [samp.pid for samp in samples]
        data_['end'] = [samp.end for samp in samples]
        data_['icu'] = [samp.icu for samp in samples]
        c_rmk = src.pid.in_(data_.pid.tolist())
        rmk_ = pd.DataFrame(list(src.select(src.rmk_t).where(c_rmk).dicts()))
        data_ = pd.concat([data_, rmk_], axis=1)
        data_ = data_.rename(columns={'rmk_t': 'rmk'})
        data_ = data_.sort_values('pid')
        data_ = data_.reset_index(drop=True)

        return data_


class FeatureLoader(Basic):
    def __init__(self, local_p: Path, save_p: Path = None):
        super().__init__()
        self.__samples = self._Basic__TableLoad(local_p)
        self.__save_p = PathVerify(save_p) if save_p else Path.cwd()

    def __GetFeatPerform(self, data_: pd.DataFrame) -> pd.DataFrame:

        row_s = []
        bin_cols = ['sex', 'gender']
        col_s = data_.columns.drop(self._Basic__info_cols).tolist()

        for col in col_s:

            tmp = data_[[self._Basic__label_col, col]]
            tmp = tmp.dropna()

            n_neg = len(tmp[tmp[self._Basic__label_col] == 0])
            n_pos = len(tmp[tmp[self._Basic__label_col] == 1])

            if n_neg < 2 or n_pos < 2:
                continue

            process = PerfomAssess(tmp[self._Basic__label_col], tmp[col])
            p_cate = 'binary' if col in bin_cols else 'continuous'
            p, rs_pos, rs_neg = process.PValueAssess(cate=p_cate)
            auc, _, _, = process.AucAssess()

            key_s = ['met', 'P', 'AUC', 'rs_0', 'size_0', 'rs_1', 'size_1']
            value_s = [col, p, auc, rs_neg, n_neg, rs_pos, n_pos]

            row = pd.Series(dict(zip(key_s, value_s)))
            row_s.append(row)

        feat = pd.DataFrame(row_s)
        feat = feat.set_index('met', drop=True)
        return feat

    def DropInfoCol(self, df: pd.DataFrame):
        df = df.drop(columns=self._Basic__info_cols)
        return df

    def VarFeatsLoad(self,
                     met_s: list = [],
                     ind_s: list = [],
                     spec_: list = [],
                     save_n: str = '') -> None:
        '''
        met_s: methods select
        ind_s: indicators select
        save_n: table save name (Default: Not Save to local)
        '''
        data_var = self._Basic__GetSampleData(self.__samples)
        met_s = met_s if met_s else self.__samples[0].data.index.to_list()
        ind_s = ind_s if ind_s else self.__samples[0].data.columns.to_list()

        for met in met_s:
            for ind in ind_s:
                col_name = met + '-' + ind
                if not spec_:
                    data_var[col_name] = [
                        samp.data.loc[met, ind] for samp in self.__samples
                    ]
                elif col_name in spec_:
                    data_var[col_name] = [
                        samp.data.loc[met, ind] for samp in self.__samples
                    ]
                else:
                    continue

        feat_var = self.__GetFeatPerform(data_var)

        if save_n:
            save_p_0 = self.__save_p / (save_n + '_data_tot.csv')
            data_var.to_csv(save_p_0, index=False)
            save_p_1 = self.__save_p / (save_n + '_feat_tot.csv')
            feat_var.to_csv(save_p_1)

        return data_var, feat_var

    def LabFeatsLoad(self, src_0: any, src_1: any, save_n: str = '') -> None:
        '''
        src_0: Patient static data
        src_1: Clinical and physiological data
        save_n: table save name (Default: Not Save to local)
        '''
        data_lab = self._Basic__GetSampleData(self.__samples)
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
        feat_lab = self.__GetFeatPerform(data_lab)

        if save_n:
            save_p_0 = self.__save_p / (save_n + '_data_tot.csv')
            data_lab.to_csv(save_p_0, index=False)
            save_p_1 = self.__save_p / (save_n + '_feat_tot.csv')
            feat_lab.to_csv(save_p_1)

        return data_lab, feat_lab


class DatasetGeneration(Basic):
    def __init__(self,
                 data_: pd.DataFrame,
                 feat_: pd.DataFrame,
                 save_p: Path = None):
        super().__init__()
        self.__data = data_
        self.__feat = feat_
        self.__save_p = PathVerify(save_p) if save_p else Path.cwd()

    @property
    def data(self):
        return self.__data

    @property
    def feat(self):
        return self.__feat

    def FeatsSelect(self, p_max: float = 0.05, specified_feats: list = []):
        if specified_feats:
            feats_all = specified_feats
        else:
            p_v_filt = self.__feat.P < p_max
            feats_all = self.__feat[p_v_filt].index
        self.__feat = self.__feat.loc[feats_all, :]
        self.__data = self.__data[[self._Basic__label_col] +
                                  self.__feat.index.to_list()]

    def DataSelect(self,
                   feat_lack_max: float = 0.4,
                   recs_lack_max: float = 0.2) -> pd.DataFrame:

        data_ = self.__data[[self._Basic__label_col] +
                            self.__feat.index.to_list()]
        recs_val = data_.isnull().sum(axis=1) < data_.shape[1] * recs_lack_max
        feat_val = data_.isnull().sum(axis=0) < data_.shape[0] * feat_lack_max
        feats = feat_val[feat_val].index.drop(self._Basic__label_col)
        feats_slt = self.__feat.loc[feats, :]
        feats_slt = feats_slt.sort_values(by=['P'], ascending=True)

        if feats_slt.empty or len(data_.end.unique()) == 1:
            data_slt = pd.DataFrame()
        else:
            data_slt = data_.loc[recs_val, feat_val]
            for feat_col in feats_slt.index:
                if data_slt[feat_col].dtype == 'bool':
                    data_slt[feat_col] = data_[feat_col].astype(int)

        self.__feat, self.__data = feats_slt, data_slt
        data_slt.to_csv(self.__save_p / 'data_slt.csv', index=False)
        feats_slt.to_csv(self.__save_p / 'feat_slt.csv')

    def FeatsDistPlot(self):
        violin_dist = self.__save_p / 'feat_violin'
        violin_dist.mkdir(parents=True, exist_ok=True)

        plot_p = PlotMain(violin_dist)
        for feat_col in self.__feats_slt.index:
            df_tmp = self.__data[[self._Basic__label_col, feat_col]]
            df_tmp['all'] = ''
            df_tmp.to_csv(violin_dist / (feat_col + '.csv'), index=False)
            plot_p.ViolinPlot(x='all',
                              y=feat_col,
                              df=df_tmp,
                              fig_n=feat_col,
                              hue=self._Basic__label_col)
