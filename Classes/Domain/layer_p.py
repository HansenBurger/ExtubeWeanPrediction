class func():
    def __init__(self) -> None:
        pass


class Patient(func):
    def __init__(self):
        super().__init__()
        self.__pid = 0
        self.__icu = ''
        self.__end_t = None
        self.__end_i = ''
        self.__rid_s = []
        self.__validy = {}
        self.__resp_l = None
        self.__para_d = None
        self.__result = None

    @property
    def pid(self):
        return self.__pid

    @pid.setter
    def pid(self, v):
        self.__pid = v

    @property
    def icu(self):
        return self.__icu

    @icu.setter
    def icu(self, v):
        self.__icu = v

    @property
    def end_t(self):
        return self.__end_t

    @end_t.setter
    def end_t(self, v):
        self.__end_t = v

    @property
    def end_i(self):
        return self.__end_i

    @end_i.setter
    def end_i(self, v):
        self.__end_i = v

    @property
    def rid_s(self):
        # string or lists
        return self.__rid_s

    @rid_s.setter
    def rid_s(self, v):
        self.__rid_s = v

    @property
    def validy(self):
        return self.__validy

    @validy.setter
    def validy(self, v):
        self.__validy = v

    @property
    def resp_l(self):
        return self.__resp_l

    @resp_l.setter
    def resp_l(self, v):
        self.__resp_l = v

    @property
    def para_d(self):
        return self.__para_d

    @para_d.setter
    def para_d(self, v):
        self.__para_d = v

    @property
    def result(self):
        return self.__result

    @result.setter
    def result(self, v):
        self.__result = v