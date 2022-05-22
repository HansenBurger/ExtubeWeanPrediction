import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, KFold


class Basic:
    def __init__(self) -> None:
        pass


class PreProcess(Basic):
    def __init__(self, df: pd.DataFrame, col_lab: str):
        super().__init__()
        self.__df_in = df
        self.__col_l = col_lab

    def DataImpute(self, impute_type: str):
        data = self.__df_in
        if impute_type == 'knn':
            imp = KNNImputer(weights='uniform')
        elif impute_type == 'mul':
            imp = IterativeImputer(max_iter=10, random_state=0)

        data_imp = imp.fit_transform(data.values.tolist())
        for i in range(data.shape[0]):
            data.loc[data.index[i]] = data_imp[i]

    def DataSplit(self, test_size_st):
        data = self.__df_in
        X = data.loc[:, data.columns != self.__col_l]
        y = data.loc[:, data.columns == self.__col_l].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_st, random_state=0)
        return X_train, y_train, X_test, y_test

    def KFoldDataSplit(self):
        pass