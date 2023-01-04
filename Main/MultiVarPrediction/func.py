import sys
import itertools
import seaborn as sns
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv, concat

sys.path.append(str(Path.cwd()))

from MLBasic.Predictors import NormPredictor, KFoldPredictor


class Basic():
    def __init__(self) -> None:
        pass


class MultiModelPredict(Basic):
    def __init__(self,
                 data_set: DataFrame,
                 algo_set: dict,
                 label_col: str,
                 save_path: Path = Path.cwd()):
        super().__init__()
        self.__data_set = data_set
        self.__algo_set = algo_set
        self.__label_col = label_col
        self.__save_path = save_path
        self.__model_result = {}

    @property
    def model_result(self):
        return self.__model_result

    def __SplitValid(self, split_n: any):
        dist_count = self.__data_set[self.__label_col].value_counts()
        dist_filt = dist_count < split_n
        validation = sum(dist_filt) == 0

        return validation

    def Basic(self, m_st: dict, save_p: Path, store_results: bool = True):
        model_p = NormPredictor(m_st['class'])
        model_p.DataSetBuild(self.__data_set, self.__label_col, m_st['split'])
        model_p.ParamSelectRand(para_pool=m_st['s_param'],
                                eval_set=m_st['eval_set'])
        model_p.Validate(para_init_add=m_st['param_init'],
                         para_deduce_add=m_st['param_deduce'],
                         re_select_feat=m_st['re_select'],
                         get_feat_imp=m_st['get_feat_imp'])
        df_rs = model_p.ResultGenerate(store_results, save_p)
        return df_rs

    def KFold(self, m_st: dict, save_p: Path, store_results: bool = True):
        if not self.__SplitValid(m_st['k-split']):
            return None
        model_p = KFoldPredictor(m_st['class'], m_st['k-split'])
        model_p.DataSetBuild(self.__data_set, self.__label_col)
        model_p.ParamSelectRand(m_st['s_param'], m_st['eval_set'])
        model_p.CrossValidate(m_st['param_init'], m_st['param_deduce'],
                              m_st['re_select'], m_st['get_feat_imp'])
        df_rs = model_p.ResultGenerate(store_results, save_p)
        df_ave = df_rs.loc['ave', :]
        return df_ave

    def MultiModels(self,
                    pred_way: str,
                    model_names: list = ['LR', 'RF', 'SVM', 'XGB'],
                    store_results: bool = True):
        for model in model_names:
            m_st = self.__algo_set[model]
            save_path = self.__save_path / model
            if pred_way == 'Norm':
                rs = self.Basic(m_st, save_path, store_results)
                self.__model_result[model] = rs
            elif pred_way == 'KFold':
                rs = self.KFold(m_st, save_path, store_results)
                self.__model_result[model] = rs


class ResultsSummary(Basic):
    def __init__(self, load_path: Path, pred_models: list) -> None:
        super().__init__()
        self.__load_p = load_path
        self.__models = pred_models
        self.__table_ = 'pred_result.csv'
        self.__cols_s = ['feat', 'sen', 'spe', 'acc', 'auc', 'f_1']

    def __LoadPerformance(self) -> dict:
        model_perform = {}
        for model in self.__models:
            rows = []
            keys = ['ord'] + self.__cols_s
            for folder in self.__load_p.iterdir():
                if not folder.is_dir() or len(folder.name.split('-')) == 1:
                    continue
                else:
                    p_ave = read_csv(folder / model / self.__table_).iloc[-1]
                    vals_0 = folder.stem.split('-', 1)
                    vals_1 = p_ave.loc[[
                        's_sen', 's_spe', 's_acc', 's_auc', 's_f_1'
                    ]].tolist()
                    row = DataFrame([dict(zip(keys, vals_0 + vals_1))])
                    rows.append(row)
            ave_df = concat(rows, ignore_index=True)
            model_perform[model] = ave_df
        return model_perform

    def TrendPlot(self) -> None:
        perform_d = self.__LoadPerformance()
        save_p = self.__load_p / '_Trend'
        save_p.mkdir(parents=True, exist_ok=True)
        for model in self.__models:
            sns.reset_orig()
            sns.set_theme(style='whitegrid')
            df = perform_d[model]
            df.ord = df.ord.astype('int')
            df.ord = df.ord + 1
            df = df.set_index('ord', drop=True)
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.plot([0, df.shape[0]], [0.7, 0.7], 'k--')
            ax.set_ylim([0.0, 1.1])
            ax.set_xlim([1, df.shape[0]])
            df_slt = df.loc[:, ['auc', 'acc', 'sen', 'spe']]
            sns.lineplot(data=df_slt, palette='tab10', linewidth=2.5, ax=ax)
            ax.set_xlabel('Number of variability indicators ($n$)',
                          fontsize=15)
            ax.set_ylabel('Assessed Value', fontsize=15)
            title_st = dict(fontname='Times New Roman',
                            size=18,
                            fontweight='bold')
            ax.set_title(model, fontdict=title_st, loc='left')
            plt.legend(loc='best')
            plt.tight_layout()
            fig.savefig(save_p / (model + '.png'), dpi=300)
            plt.close()

    def BestDisplay(self, d_col_n: str = 'auc', imp_min: float = 0.8):
        '''
        Save the best performance of foward search
        :param: d_col_n: best depend column name
        :param: imp_min: imp min shape ratio
        '''
        perform_d = self.__LoadPerformance()

        for model in self.__models:
            save_p = self.__load_p / '_Best' / model
            save_p.mkdir(parents=True, exist_ok=True)
            df_sort = perform_d[model].sort_values(by=[d_col_n],
                                                   ascending=False)
            df_sort = df_sort.reset_index()
            df_best = df_sort.loc[0, ['ord', 'feat']]
            load_n = '-'.join(df_best.tolist())
            load_p = self.__load_p / load_n / model
            df_rs = read_csv(load_p / self.__table_)
            df_rs = df_rs.round(3)
            df_rs_n = '_'.join([load_n, 'Performance'])
            df_rs.to_csv(save_p / (df_rs_n + '.csv'), index=False)

            min_feat = int(df_best['ord']) * imp_min
            imp_df_s = []
            for file in load_p.iterdir():
                if file.suffix != '.csv':
                    continue
                elif 'Imp' not in file.name:
                    continue
                else:
                    imp_df = read_csv(file, index_col=0)
                    col_map = dict(
                        zip(imp_df.columns.to_list(),
                            [file.name.split('_')[0]]))
                    imp_df = imp_df.rename(columns=col_map)
                    if imp_df.shape[0] >= min_feat:
                        imp_df_s.append(imp_df)

            if not imp_df_s:
                continue

            imp_feat = list(
                itertools.chain.from_iterable(
                    [df.index.tolist() for df in imp_df_s]))
            dup_feat = [
                k for k, v in Counter(imp_feat).items() if v == len(imp_df_s)
            ]
            imp_df_s = [df.loc[dup_feat, :] for df in imp_df_s]
            imp_df_s = [df.sort_index() for df in imp_df_s]
            imp_df = concat(imp_df_s, axis=1)
            imp_df['ave'] = imp_df.mean(numeric_only=True, axis=1)
            imp_df['med'] = imp_df.median(numeric_only=True, axis=1)

            imp_df_n = '_'.join([load_n, 'Importance'])
            imp_df.to_csv(save_p / (imp_df_n + '.csv'))
            dim_st = (12, 0.6 * imp_df.shape[0])
            fig, (ax_0,
                  ax_1) = plt.subplots(1,
                                       2,
                                       figsize=dim_st,
                                       gridspec_kw={'width_ratios': [1, 1]})

            xstick_st = dict(fontname='Times New Roman', size=15)

            ave_sort = imp_df.sort_values(by=['ave'], ascending=False)
            sns.barplot(y=ave_sort.index, x=ave_sort.ave, ax=ax_0)
            ax_0.set_ylabel('Breathing Variability', fontsize=15)
            ax_0.set_xlabel('(a) Average', fontdict=xstick_st)

            med_sort = imp_df.sort_values(by=['med'], ascending=False)
            sns.barplot(y=med_sort.index, x=med_sort.med, ax=ax_1)
            # ax_1.set_ylabel('Breathing Variability', fontsize=15)
            ax_1.set_xlabel('(b) Median', fontdict=xstick_st)
            plt.tight_layout()
            fig.savefig(save_p / (imp_df_n + '.png'), dpi=300)


# p_l = [
#     'C:\\Main\\Data\\_\\Result\\Mix\\20220807_09_ForwardSearch_gp_0_basic_Nvar\\Extube_SumP12_Nad-60\\Graph',
#     'C:\\Main\\Data\\_\\Result\\Mix\\20220807_10_ForwardSearch_gp_3_inds_Nvar\\Extube_SumP12_Nad-60\\Graph',
#     'C:\\Main\\Data\\_\\Result\\Mix\\20220807_11_ForwardSearch_gp_6_mets_Nvar\\Extube_SumP12_Nad-60\\Graph',
#     'C:\\Main\\Data\\_\\Result\\Mix\\20220807_14_ForwardSearch_gp_8_all_Nvar\\Extube_SumP12_Nad-60\\Graph'
# ]
# models = ['LR', 'RF', 'SVM', 'XGB']
# for p in p_l:
#     main_p = ResultsSummary(Path(p), models)
#     main_p.BestDisplay()