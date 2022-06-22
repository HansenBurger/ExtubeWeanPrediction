import warnings
import pandas as pd

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')


class ModelBasic:
    def __init__(self, algorithm):
        self.__model = None
        self.__dataset = {}
        self.__algorithm = algorithm

    @property
    def dataset(self):
        return self.__dataset

    @property
    def algorithm(self):
        return self.__algorithm

    def DataInit(self, X_train, y_train, X_test, y_test) -> None:
        self.__dataset['X_train'] = X_train
        self.__dataset['y_train'] = y_train
        self.__dataset['X_test'] = X_test
        self.__dataset['y_test'] = y_test

    def ModelInit(self, param: dict) -> None:
        '''
        Initializing the model with model classes and parameters
        model_cls: any model class

        '''
        self.__model = self.__algorithm(**param)

    def Deduce(self, param: dict = {}) -> None:
        '''
        Use traning data to deduce the model
        '''
        X_in = self.__dataset['X_train']
        y_in = self.__dataset['y_train']

        self.__model.fit(X_in, y_in, **param)

    def Predict(self) -> dict:
        '''
        Use test data to estimate the model
        '''
        X_in = self.__dataset['X_test']
        y_in = self.__dataset['y_test']

        pred_l = self.__model.predict(X_in)
        pred_p = self.__model.predict_proba(X_in)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_in, pred_l, labels=[0, 1]).ravel()
        sens = tp / (tp + fn)
        spec = tn / (fp + tn)
        r2 = round(self.__model.score(X_in, y_in), 4)
        f1 = round(f1_score(y_in, pred_l, average='binary'), 4)
        auc = round(roc_auc_score(y_in, pred_p), 4)
        report = classification_report(y_in, pred_l)

        perform_rs = {
            'label': pred_l,
            'prob': pred_p,
            'f1': f1,
            'r2': r2,
            'sens': sens,
            'spec': spec,
            'auc': auc,
            'report': report,
        }

        return perform_rs


class LogisiticReg(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(LogisticRegression)
        self.DataInit(*data_)
        self.ModelInit(s_param)


class RandomForest(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(RandomForestClassifier)
        self.DataInit(*data_)
        self.ModelInit(s_param)


class SupportVector(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(SVC)
        self.DataInit(*data_)
        self.ModelInit(s_param)


class ParaSel_Grid(ModelBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__(GridSearchCV)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def BestParam(self) -> dict:
        best_param = self._ModelBasic__model.best_params_
        return best_param


class ParaSel_Rand(ModelBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__(RandomizedSearchCV)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def BestParam(self) -> dict:
        best_param = self._ModelBasic__model.best_params_
        return best_param


class XGBoosterClassify(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(XGBClassifier)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def GetFeatureImportance(self) -> pd.Series:
        data_set = self._ModelBasic__dataset
        attr_imp = pd.Series(
            self._ModelBasic__model.get_booster().get_fscore())
        attr_select = attr_imp.loc[attr_imp > 0].index

        if attr_select.empty:
            pass
        else:
            self._ModelBasic__dataset['X_train'] = data_set['X_train'][
                attr_select]
            self._ModelBasic__dataset['X_test'] = data_set['X_test'][
                attr_select]

        return attr_imp
