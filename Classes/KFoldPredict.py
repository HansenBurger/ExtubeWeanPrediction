import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain
from Classes.MLD.balancefunc import BalanSMOTE
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

    def DataSetBuild(self, data_, col_label):

        # Data split
        main_p = DataSetProcess(data_, col_label)
        main_p.DataImpute('knn')
        data_l = main_p.KFoldSplit(self.__fold_num)

        # data balance (SMOTE)
        for data_ in data_l:
            balance_p = BalanSMOTE(data_[0], data_[1])
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

                para_deduce['eval_set'] = self.__GetEvalSet(main_p.dataset)
                para_deduce['early_stopping_rounds'] = 50
                main_p.Deduce(para_deduce)
                self.__fold_pred.append(main_p.Predict())

    def ResultGenerate(self, save_path: Path):
        repr_gen = lambda dict_: ('\n').join(k + ':\t' + str(v)
                                             for k, v in dict_.items())

        with open(save_path / 'pred_result.txt', 'w') as f:
            for i in range(self.__fold_num):
                fold_info = self.__fold_pred[i]
                fold_para = self.__fold_para[i]

                f.write('\n{0}-Fold:\n'.format(i))
                f.write('FSCORE: \t {0} \n'.format(fold_info['score']))
                f.write('ROCAUC: \t {0} \n'.format(fold_info['rocauc']))
                f.write('REPORT: \n {0} \n'.format(fold_info['report']))
                f.write('PARAMS: \n {0} \n'.format(repr_gen(fold_para)))

            ave_auc = sum([fold['rocauc']
                           for fold in self.__fold_pred]) / self.__fold_num
            ave_score = sum([fold['score']
                             for fold in self.__fold_pred]) / self.__fold_num
            f.write('\nAVE Performance:\n')
            f.write('AUC:\t{0}\n'.format(ave_auc))
            f.write('F1SCORE:\t{0}\n'.format(ave_score))

        for i in range(self.__fold_num):
            fold_info = self.__fold_pred[i]
            fold_data = self.__fold_data[i]
            save_name = '{0}-Fold_ROC.png'.format(i)
            main_p = PlotMain(save_path)
            main_p.RocSinglePlot(fold_data[3], fold_info['prob'], save_name)
