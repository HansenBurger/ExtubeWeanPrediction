class func():
    def __init__(self) -> None:
        pass


class RecWave(func):
    def __init__(self):
        super().__init__()
        self.__zdt = ''
        self.__rec_t = None
        self.__sr = -1
        self.__p_ind = []
        self.__s_ind = []
        self.__e_ind = []
        self.__s_P = []
        self.__s_F = []
        self.__s_V = []
        self.__resps = []

    @property
    def zdt(self):
        return self.__zdt

    @zdt.setter
    def zdt(self, v):
        self.__zdt = v

    @property
    def rec_t(self):
        return self.__rec_t

    @rec_t.setter
    def rec_t(self, v):
        self.__rec_t = v

    @property
    def sr(self):
        return self.__sr

    @sr.setter
    def sr(self, v):
        self.__sr = v

    @property
    def p_ind(self):
        return self.__p_ind

    @p_ind.setter
    def p_ind(self, v):
        self.__p_ind = v

    @property
    def s_ind(self):
        return self.__s_ind

    @s_ind.setter
    def s_ind(self, v):
        self.__s_ind = v

    @property
    def e_ind(self):
        return self.__e_ind

    @e_ind.setter
    def e_ind(self, v):
        self.__e_ind = v

    @property
    def s_P(self):
        return self.__s_P

    @s_P.setter
    def s_P(self, v):
        self.__s_P = v

    @property
    def s_F(self):
        return self.__s_F

    @s_F.setter
    def s_F(self, v):
        self.__s_F = v

    @property
    def s_V(self):
        return self.__s_V

    @s_V.setter
    def s_V(self, v):
        self.__s_V = v

    @property
    def resps(self):
        return self.__resps

    @resps.setter
    def resps(self, v):
        self.__resps = v


class RecPara(func):
    def __init__(self):
        super().__init__()
        self.__zpx = ''
        self.__u_ind = []
        self.__st_mode = None  # pre save dict, then num
        self.__st_peep = []
        self.__st_ps = []
        self.__st_e_sens = []
        self.__st_sump = []
        self.__bed_hr = []
        self.__bed_sbp = []
        self.__bed_dbp = []
        self.__bed_mbp = []
        self.__bed_spo2 = []
        self.__bed_rr = []
        self.__bed_pr = []
        self.__bed_cvpm = []

    @property
    def zpx(self):
        return self.__zpx

    @zpx.setter
    def zpx(self, v):
        self.__zpx = v

    @property
    def u_ind(self):
        return self.__u_ind

    @u_ind.setter
    def u_ind(self, v):
        self.__u_ind = v

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, v):
        self.__mode = v

    @property
    def st_mode(self):
        return self.__st_mode

    @st_mode.setter
    def st_mode(self, v):
        self.__st_mode = v

    @property
    def st_peep(self):
        return self.__st_peep

    @st_peep.setter
    def st_peep(self, v):
        self.__st_peep = v

    @property
    def st_ps(self):
        return self.__st_ps

    @st_ps.setter
    def st_ps(self, v):
        self.__st_ps = v

    @property
    def st_e_sens(self):
        return self.__st_e_sens

    @st_e_sens.setter
    def st_e_sens(self, v):
        self.__st_e_sens = v

    @property
    def st_sump(self):
        return self.__st_sump

    @st_sump.setter
    def st_sump(self, v):
        self.__st_sump = v

    @property
    def bed_hr(self):
        return self.__bed_hr

    @bed_hr.setter
    def bed_hr(self, v):
        self.__bed_hr = v

    @property
    def bed_sbp(self):
        return self.__bed_sbp

    @bed_sbp.setter
    def bed_sbp(self, v):
        self.__bed_sbp = v

    @property
    def bed_dbp(self):
        return self.__bed_dbp

    @bed_dbp.setter
    def bed_dbp(self, v):
        self.__bed_dbp = v

    @property
    def bed_mbp(self):
        return self.__bed_mbp

    @bed_mbp.setter
    def bed_mbp(self, v):
        self.__bed_mbp = v

    @property
    def bed_spo2(self):
        return self.__bed_spo2

    @bed_spo2.setter
    def bed_spo2(self, v):
        self.__bed_spo2 = v

    @property
    def bed_rr(self):
        return self.__bed_rr

    @bed_rr.setter
    def bed_rr(self, v):
        self.__bed_rr = v

    @property
    def bed_pr(self):
        return self.__bed_pr

    @bed_pr.setter
    def bed_pr(self, v):
        self.__bed_pr = v

    @property
    def bed_cvpm(self):
        return self.__bed_cvpm

    @bed_cvpm.setter
    def bed_cvpm(self, v):
        self.__bed_cvpm = v


class DomainTS(func):
    def __init__(self) -> None:
        super().__init__()
        self.__ave = None
        self.__med = None
        self.__std = None
        self.__cv = None
        self.__qua = None
        self.__tqua = None

    @property
    def ave(self):
        return self.__ave

    @ave.setter
    def ave(self, obj):
        self.__ave = obj

    @property
    def med(self):
        return self.__med

    @med.setter
    def med(self, obj):
        self.__med = obj

    @property
    def std(self):
        return self.__std

    @std.setter
    def std(self, obj):
        self.__std = obj

    @property
    def cv(self):
        return self.__cv

    @cv.setter
    def cv(self, obj):
        self.__cv = obj

    @property
    def qua(self):
        return self.__qua

    @qua.setter
    def qua(self, obj):
        self.__qua = obj

    @property
    def tqua(self):
        return self.__tqua

    @tqua.setter
    def tqua(self, obj):
        self.__tqua = obj


class DomainFS(func):
    def __init__(self):
        super().__init__()
        self.__prsa = None

    @property
    def prsa(self):
        return self.__prsa


class DomainHRV(func):
    def __init__(self):
        super().__init__()
        self.__sd1 = None
        self.__sd2 = None

    @property
    def sd1(self):
        return self.__sd1

    @sd1.setter
    def sd1(self, obj):
        self.__sd1 = obj

    @property
    def sd2(self):
        return self.__sd2

    @sd2.setter
    def sd2(self, obj):
        self.__sd2 = obj


class DomainHRA(func):
    def __init__(self):
        super().__init__()
        self.__pi = None
        self.__gi = None
        self.__si = None

    @property
    def pi(self):
        return self.__pi

    @pi.setter
    def pi(self, obj):
        self.__pi = obj

    @property
    def gi(self):
        return self.__gi

    @gi.setter
    def gi(self, obj):
        self.__gi = obj

    @property
    def si(self):
        return self.__si

    @si.setter
    def si(self, obj):
        self.__si = obj