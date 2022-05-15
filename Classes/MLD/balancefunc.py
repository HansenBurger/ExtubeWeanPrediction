from numpy import unique
from imblearn.over_sampling import SMOTE


class Basic:
    def __init__(self) -> None:
        pass

    def ArrayUniqueCounter(self, a):
        u_i, u_c = unique(a, return_counts=True)
        u_d = dict(zip(u_i, u_c))
        return u_d


class SMOTE_(Basic):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.__os = SMOTE(random_state=0)
        self.__X = X
        self.__y = y

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    def __repr__(self):
        '''
        Count the class size before or after SMOTE(Limit 2 classification)
        '''
        au = self.ArrayUniqueCounter(self.__y)
        ratic_c = lambda x, y: round(y[x] / sum(y.values()), 2)
        neg, pos = au[0], au[1]
        n_r, p_r = ratic_c(0, au), ratic_c(1, au)
        sr = 'Neg:Pos = {0}:{1}, ratio = {2}:{3}'.format(neg, pos, n_r, p_r)
        return sr

    def OverSample(self):
        '''
        ReSampleStrategies: auto(minor oversampling; major undersampling)
        '''
        self.__X, self.__y = self.__os.fit_resample(self.__X, self.__y)
        #TODO data format switching process
