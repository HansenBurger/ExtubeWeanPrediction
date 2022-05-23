import pandas as pd

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


class SKLBasic:
    def __init__(self):
        self.__model = None
        self.__dataset = {}

    @property
    def dataset(self):
        return self.__dataset

    def DataInit(self, X_train, y_train, X_test, y_test) -> None:
        self.__dataset['X_train'] = X_train
        self.__dataset['y_train'] = y_train
        self.__dataset['X_test'] = X_test
        self.__dataset['y_test'] = y_test

    def ModelInit(self, model_cls: any, param: dict) -> None:
        '''
        Initializing the model with model classes and parameters
        model_cls: any model class

        '''
        self.__model = model_cls(**param)

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
        score = round(self.__model.score(X_in, y_in), 2)
        report = classification_report(y_in, pred_l)
        rocauc = roc_auc_score(y_in, pred_p)

        perform_rs = {
            'label': pred_l,
            'prob': pred_p,
            'score': score,
            'report': report,
            'rocauc': rocauc
        }

        return perform_rs


class LogisiticReg(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(LogisticRegression, s_param)


class RandomForest(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(RandomForestClassifier, s_param)


class SupportVector(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(SVC, s_param)


class ParaSel_Grid(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(GridSearchCV, s_param)

    def BestParam(self) -> dict:
        best_param = self._SKLBasic__model.best_param_
        return best_param


class ParaSel_Rand(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(RandomizedSearchCV, s_param)

    def BestParam(self) -> dict:
        best_param = self._SKLBasic__model.best_param_
        return best_param


class XGBoosterClassify(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(XGBClassifier, s_param)

    def GetFeatureImportance(self) -> pd.Series:
        data_set = self._SKLBasic__dataset
        attr_imp = pd.Series(self._SKLBasic__model.get_booster().get_fscore())
        attr_select = attr_imp.loc[attr_imp > 0].index

        data_set['X_train'] = data_set['X_train'][attr_select]
        data_set['X_test'] = data_set['X_test'][attr_select]

        return attr_imp
