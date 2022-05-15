import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class Basic:
    def __init__(self) -> None:
        pass


class PreProcess(Basic):
    def __init__(self, col_label, file_loc):
        super().__init__()
        self.__col_l = col_label
        self.__df = pd.read_csv(file_loc)
        self.__ds = {}

    @property
    def df(self):
        return self.__df

    @property
    def ds(self):
        return self.__ds

    def FeatureFilt(self, targets, methods):
        col_n_s = [x + '_' + y for y in methods for x in targets]
        self.__df = self.__df[[self.__col_l] + col_n_s]

    def DataSplit(self, test_size_st):
        data = self.__df
        X = data.loc[:, data.columns != self.__col_l]
        y = data.loc[:, data.columns == self.__col_l].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_st, random_state=0)
        self.__ds = {
            'train': {
                'X': X_train,
                'y': y_train
            },
            'test': {
                'X': X_test,
                'y': y_test
            }
        }


class ResultGen(Basic):
    def __init__(self, save_loc, file_name) -> None:
        super().__init__()
        self.__folder = save_loc
        self.__name = file_name

    def TextCollect(self, print_info):
        file_loc = self.__folder / (self.__name + '.txt')
        with open(file_loc, 'w') as f:
            f.write(print_info)
            f.write('\n')

    def GraphCollect(self, true_a, pred_a):
        file_loc = self.__folder / (self.__name + '.png')
        roc_auc = roc_auc_score(true_a, pred_a)
        fpr, tpr, thresholds = roc_curve(true_a, pred_a)
        fig_dims = (6, 6)
        plt.subplots(figsize=fig_dims)
        plt.plot(fpr, tpr, label='ROC-AUC (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(file_loc)
        plt.close()