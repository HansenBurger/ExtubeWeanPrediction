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
        self.__predict_rd = {}
        self.__perform_rd = {}

    @property
    def dataset(self):
        return self.__dataset

    @property
    def predict_rd(self):
        return self.__predict_rd

    @property
    def perform_rd(self):
        return self.__perform_rd

    def DataInit(self, X_train, y_train, X_test, y_test):
        self.__dataset['X_train'] = X_train
        self.__dataset['y_train'] = y_train
        self.__dataset['X_test'] = X_test
        self.__dataset['y_test'] = y_test

    def ModelInit(self, model_cls: any, param: dict):
        '''
        Initializing the model with model classes and parameters
        model_cls: any model class

        '''
        self.__model = model_cls(**param)

    def Deduce(self, X_in, y_in, param: dict = {}):
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        self.__model.fit(X_in, y_in, **param)

    def Predict(self, X_in, y_in):
        '''
        Use test data to estimate the model
        X_in: X_test
        y_in: y_test
        '''
        pred_l = self.__model.predict(X_in)
        pred_p = self.__model.predict_proba(X_in)[:, 1]
        score = round(self.__model.score(X_in, y_in), 2)
        report = classification_report(y_in, pred_l)
        rocauc = roc_auc_score(y_in, pred_p)
        self.__predict_rd = {'label': pred_l, 'prob': pred_p}
        self.__perform_rd = {
            'score': score,
            'report': report,
            'rocauc': rocauc
        }


class LogisiticReg(SKLBasic):
    def __init__(self, data_: list, s_param: dict):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(LogisticRegression, **s_param)


class RandomForest(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(RandomForestClassifier, **s_param)


class SupportVector(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.DataInit(*data_)
        self.ModelInit(SVC, **s_param)


class XGBoosterClassify(SKLBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__()
        self.__attr_imp = pd.Series()
        self.DataInit(*data_)
        self.ModelInit(XGBClassifier, **s_param)

    @property
    def attr_imp(self):
        return self.__attr_imp

    def GetFeatureImportance(self):
        '''
        
        '''
        self.__attr_imp = pd.Series(
            self.__SKLBasic_model.get_booster().get_fscore())
        attr_select = self.__attr_imp.loc[self.__attr_imp > 0].index
        X_train = self.dataset['X_train'][attr_select]
        X_test = self.dataset['X_test'][attr_select]

        return X_train, X_test


class XGBoosterClassify_(SKLBasic):
    def __init__(self, s_param: dict = {}):
        super().__init__()
        self.__model = XGBClassifier(**s_param)
        self.predict_d = {}
        self.perform_d = {}

    def Deduce(self, X_in, y_in, eval_set: list(tuple) = None) -> None:
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        # eval set for early stop
        self.__model.fit(X_in,
                         y_in,
                         eval_set=eval_set,
                         eval_metric='auc',
                         early_stopping_rounds=10,
                         verbose=True)


class ParamsSelect():
    def __init__(self, s_param):
        self.__grid_sch_cv = GridSearchCV(**s_param)
        self.__rand_sch_cv = RandomizedSearchCV(**s_param)

    def Deduce(self, sch_type: str = 'rand'):
        pass