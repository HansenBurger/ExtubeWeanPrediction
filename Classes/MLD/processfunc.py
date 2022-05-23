import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, KFold


class Basic:
    def __init__(self) -> None:
        pass


class DataSetProcess(Basic):
    def __init__(self, df: pd.DataFrame, col_lab: str):
        super().__init__()
        self.__data = df
        self.__coln = col_lab

    def __GetXy(self):
        data_ = self.__data
        X = data_.loc[:, data_.columns != self.__coln]
        y = data_.loc[:, data_.columns == self.__coln].values.ravel()
        return X, y

    def DataImpute(self, impute_type: str = 'knn'):
        data_ = self.__data

        if impute_type == 'knn':
            imp = KNNImputer(weights='uniform')
        elif impute_type == 'mul':
            imp = IterativeImputer(max_iter=10, random_state=0)

        data_imp = imp.fit_transform(data_.values.tolist())
        for i in range(data_.shape[0]):
            data_.loc[data_.index[i]] = data_imp[i]

    def DataSplit(self, test_size_st):
        X, y = self.__GetXy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_st, random_state=0)
        return X_train, y_train, X_test, y_test

    def KFoldSplit(self, split_n: int = 5, rand_n: int = 104):
        data_l = []
        X, y = self.__GetXy()
        kf = KFold(n_splits=split_n,
                   random_state=rand_n,
                   shuffle=True if rand_n > 0 else False)

        for train_i, test_i in kf.split(X):
            X_train, y_train = X[train_i], y[train_i]
            X_test, y_test = y[test_i], y[test_i]
            data_l.append([X_train, y_train, X_test, y_test])

        return data_l
