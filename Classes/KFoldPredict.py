import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain
from Classes.MLD.balancefunc import BalanSMOTE, BalanRandOS
from Classes.MLD.processfunc import DataSetProcess
from Classes.MLD.algorithm import ParaSel_Grid, ParaSel_Rand


class Basic():
    def __init__(self):
        pass


class KFoldMain(Basic):
    def __init__(self, algo_class: any, split_n: int = 5) -> None:
        super().__init__()
        self.__algo_type = algo_class
        self.__fold_num = split_n
        self.__fold_data = []
        self.__fold_para = []
        self.__fold_pred = []
        self.__ave_result = {}

    @property
    def ave_result(self):
        return self.__ave_result

    def DataSetBuild(self, data_, col_label):

        # Data split
        main_p = DataSetProcess(data_, col_label)
        main_p.DataImpute('knn')
        data_l = main_p.KFoldSplit(self.__fold_num)

        # data balance (SMOTE)
        for data_ in data_l:
            try:
                balance_p = BalanSMOTE(data_[0], data_[1])
                balance_p.OverSample()
                data_[0], data_[1] = balance_p.X, balance_p.y
            except:
                balance_p = BalanRandOS(data_[0], data_[1])
                balance_p.OverSample()
                data_[0], data_[1] = balance_p.X, balance_p.y

            self.__fold_data.append(data_)

    def __GetEvalSet(self, data_set):
        eval_set = [(data_set['X_train'], data_set['y_train']),
                    (data_set['X_test'], data_set['y_test'])]
        return eval_set

    def ParamSelectRand(self, para_pool: dict, eval_set: bool = False):

        para_init = {
            'estimator': self.__algo_type().algorithm(),
            'param_distributions': para_pool,
            'scoring': 'accuracy',
            'cv': 3,
        }

        for i in range(self.__fold_num):
            main_p = ParaSel_Rand(self.__fold_data[i], para_init)
            para_deduce = {
                'eval_set': self.__GetEvalSet(main_p.dataset),
                'early_stopping_rounds': 10,
                'verbose': True
            } if eval_set else {}
            main_p.Deduce(para_deduce)
            self.__fold_para.append(main_p.BestParam())

    def CrossValidate(self,
                      para_init_add: dict = {},
                      para_deduce_add: dict = {},
                      re_select_feat: bool = False):

        for i in range(self.__fold_num):
            para_init = self.__fold_para[i]
            para_init.update(para_init_add)
            main_p = self.__algo_type(self.__fold_data[i], para_init)

            if not re_select_feat:
                para_deduce = {}
                para_deduce.update(para_deduce_add)
                main_p.Deduce(para_deduce)
                self.__fold_pred.append(main_p.Predict())
            else:
                para_deduce = {
                    'eval_set': self.__GetEvalSet(main_p.dataset),
                    'early_stopping_rounds': 10,
                    'eval_metric': 'auc',
                    'verbose': True
                }
                para_deduce.update(para_deduce_add)
                main_p.Deduce(para_deduce)
                _ = main_p.GetFeatureImportance()

                # if main_p.dataset['X_train']

                para_deduce['eval_set'] = self.__GetEvalSet(main_p.dataset)
                para_deduce['early_stopping_rounds'] = 50
                main_p.Deduce(para_deduce)
                self.__fold_pred.append(main_p.Predict())

    def ResultGenerate(self, save_path: Path):
        repr_gen = lambda dict_: ('\n').join(k + ':\t' + str(v)
                                             for k, v in dict_.items())

        ave_auc = sum([fold['auc']
                       for fold in self.__fold_pred]) / self.__fold_num
        ave_f1 = sum([fold['f1']
                      for fold in self.__fold_pred]) / self.__fold_num
        ave_r2 = sum([fold['r2']
                      for fold in self.__fold_pred]) / self.__fold_num
        ave_sens = sum([fold['sens']
                        for fold in self.__fold_pred]) / self.__fold_num
        ave_spec = sum([fold['spec']
                        for fold in self.__fold_pred]) / self.__fold_num

        self.__ave_result = {
            'auc': ave_auc,
            'f1': ave_f1,
            'r2': ave_r2,
            'sens': ave_sens,
            'spec': ave_spec
        }

        with open(save_path / 'pred_result.txt', 'w') as f:
            for i in range(self.__fold_num):
                fold_info = self.__fold_pred[i]
                fold_para = self.__fold_para[i]

                f.write('\n{0}-Fold:\n'.format(i))
                f.write('ROCAUC: \t {0} \n'.format(fold_info['auc']))
                f.write('F1-SCORE: \t {0} \n'.format(fold_info['f1']))
                f.write('R2-SCORE: \t {0} \n'.format(fold_info['r2']))
                f.write('REPORT: \n {0} \n'.format(fold_info['report']))
                f.write('PARAMS: \n {0} \n'.format(repr_gen(fold_para)))

            f.write('\nAVE Performance:\n')
            f.write('SCORE:\t{0}\n'.format(ave_r2))
            f.write('ROCAUC:\t{0}\n'.format(ave_auc))

        for i in range(self.__fold_num):
            fold_info = self.__fold_pred[i]
            fold_data = self.__fold_data[i]
            save_name = '{0}-Fold_ROC.png'.format(i)
            main_p = PlotMain(save_path)
            main_p.RocSinglePlot(fold_data[3], fold_info['prob'], save_name)

        pred_df = pd.DataFrame()
        pred_df['mode'] = [
            'fold_' + str(i + 1) for i in range(self.__fold_num)
        ] + ['ave']
        pred_df['auc'] = [round(i['auc'], 3)
                          for i in self.__fold_pred] + [round(ave_auc, 3)]
        pred_df['f1'] = [round(i['f1'], 3)
                         for i in self.__fold_pred] + [round(ave_f1, 3)]
        pred_df['r2'] = [round(i['r2'], 3)
                         for i in self.__fold_pred] + [round(ave_r2, 3)]
        pred_df['sens'] = [round(i['sens'], 3)
                           for i in self.__fold_pred] + [round(ave_sens, 3)]
        pred_df['spec'] = [round(i['spec'], 3)
                           for i in self.__fold_pred] + [round(ave_spec, 3)]

        pred_df.set_index('mode', drop=True)
        pd.DataFrame.to_csv(pred_df,
                            save_path / 'pred_result.csv',
                            index=False)