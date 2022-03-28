class func():
    def __init__(self) -> None:
        pass


class RidRec(func):
    def __init__(self):
        super().__init__()
        self.__zif = ''
        self.__vm_n = ''
        self.__cls_t = None
        self.__waves = []
        self.__paras = []

    @property
    def zif(self):
        return self.__zif

    @zif.setter
    def zif(self, v):
        self.__zif = v

    @property
    def vm_n(self):
        return self.__vm_n

    @vm_n.setter
    def vm_n(self, v):
        self.__vm_n = v

    @property
    def cls_t(self):
        return self.__cls_t

    @cls_t.setter
    def cls_t(self, v):
        self.__cls_t = v

    @property
    def waves(self):
        return self.__waves

    @waves.setter
    def waves(self, obj):
        self.__waves = obj

    @property
    def paras(self):
        return self.__paras

    @paras.setter
    def paras(self, obj):
        self.__paras = obj


class Result(func):
    def __init__(self):
        super().__init__()
        self.__td = None
        self.__fd = None
        self.__hra = None
        self.__hrv = None

    @property
    def td(self):
        return self.__td

    @td.setter
    def td(self, obj):
        self.__td = obj

    @property
    def fd(self):
        return self.__fd

    @fd.setter
    def fd(self, obj):
        self.__fd = obj

    @property
    def hra(self):
        return self.__hra

    @hra.setter
    def hra(self, obj):
        self.__hra = obj

    @property
    def hrv(self):
        return self.__hrv

    @hrv.setter
    def hrv(self, obj):
        self.__hrv = obj
