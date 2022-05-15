import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Basic:
    def __init__(self) -> None:
        pass


class LogisiticReg(Basic):
    def __init__(self, param_dict=None):
        super().__init__()
        self.__model = LogisticRegression(**param_dict)
        self.predict_d = None
        self.perform_d = None

    def Deduce(self, X_in, y_in):
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        self.__model.fit(X_in, y_in)

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
        self.predict_d = {'label': pred_l, 'prob': pred_p}
        self.perform_d = {'score': score, 'report': report, 'rocauc': rocauc}


class RandomForest(Basic):
    def __init__(self, param_dict=None):
        super().__init__()
        self.__model = RandomForestClassifier(**param_dict)
        self.predict_d = None
        self.perform_d = None

    def Deduce(self, X_in, y_in):
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        self.__model.fit(X_in, y_in)

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
        self.predict_d = {'label': pred_l, 'prob': pred_p}
        self.perform_d = {'score': score, 'report': report, 'rocauc': rocauc}


class SupportVector():
    def __init__(self, param_dict=None):
        super().__init__()
        self.__model = SVC(**param_dict)
        self.predict_d = None
        self.perform_d = None

    def Deduce(self, X_in, y_in):
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        self.__model.fit(X_in, y_in)

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
        self.predict_d = {'label': pred_l, 'prob': pred_p}
        self.perform_d = {'score': score, 'report': report, 'rocauc': rocauc}


class KFold(Basic):
    def __init__(self, param_dict):
        super().__init__()
        self.__model = GridSearchCV(**param_dict)
        self.predict_d = None
        self.perform_d = None

    def Deduce(self, X_in, y_in):
        '''
        Use traning data to deduce the model
        X_in: X_train
        y_in: y_train
        '''
        self.__model.fit(X_in, y_in)

    def ParaSelection(self):
        best_estimator = self.model.best_estimator_
        return best_estimator

    def Predict(self, X_in, y_in):
        pred_l = self.__model.predict(X_in)
        try:
            pred_p = self.__model.predict_proba(X_in)[:, 1]
        except:
            pred_p = self.__model.decision_function(X_in)
        pass
        best_estimator = self.__model.best_estimator_
        score = round(self.__model.score(X_in, y_in), 2)
        report = classification_report(y_in, pred_l)
        rocauc = roc_auc_score(y_in, pred_p)
        self.predict_d = {'label': pred_l, 'prob': pred_p}
        self.perform_d = {
            'score': score,
            'report': report,
            'rocauc': rocauc,
            'estimator': best_estimator
        }
