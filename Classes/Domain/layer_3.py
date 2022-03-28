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
        self.__rid = []
        self.__result = None

    @property
    def pid(self):
        return self.__pid

    @property
    def icu(self):
        return self.__icu

    @property
    def end_t(self):
        return self.__end_t

    @property
    def end_i(self):
        return self.__end_i

    @property
    def rid(self):
        return self.__rid

    @property
    def result(self):
        return self.__result