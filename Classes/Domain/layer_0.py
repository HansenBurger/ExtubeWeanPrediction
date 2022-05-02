class func():
    def __init__(self) -> None:
        pass


class Resp(func):
    def __init__(self):
        super().__init__()
        self.__wid = None
        self.__val = None
        self.__pip = 0.0
        self.__mpaw = 0.0
        self.__rr = 0.0
        self.__v_t_i = 0.0
        self.__v_t_e = 0.0
        self.__ve = 0.0
        self.__rsbi = 0.0
        self.__wob_a = 0.0
        self.__wob_b = 0.0
        self.__wob_f = 0.0
        self.__wob = 0.0
        self.__mp_jm_d = 0.0
        self.__mp_jl_d = 0.0
        self.__mp_jm_t = 0.0
        self.__mp_jl_t = 0.0

    @property
    def wid(self):
        return self.__wid

    @wid.setter
    def wid(self, v):
        self.__wid = v

    @property
    def val(self):
        return self.__val

    @val.setter
    def val(self, v):
        self.__val = v

    @property
    def pip(self):
        return self.__pip

    @pip.setter
    def pip(self, v):
        self.__pip = v

    @property
    def mpaw(self):
        return self.__mpaw

    @mpaw.setter
    def mpaw(self, v):
        self.__mpaw = v

    @property
    def rr(self):
        return self.__rr

    @rr.setter
    def rr(self, v):
        self.__rr = v

    @property
    def v_t_i(self):
        return self.__v_t_i

    @v_t_i.setter
    def v_t_i(self, v):
        self.__v_t_i = v

    @property
    def v_t_e(self):
        return self.__v_t_e

    @v_t_e.setter
    def v_t_e(self, v):
        self.__v_t_e = v

    @property
    def ve(self):
        return self.__ve

    @ve.setter
    def ve(self, v):
        self.__ve = v

    @property
    def wob(self):
        return self.__wob

    @wob.setter
    def wob(self, v):
        self.__wob = v

    @property
    def wob_a(self):
        return self.__wob_a

    @wob_a.setter
    def wob_a(self, v):
        self.__wob_a = v

    @property
    def wob_f(self):
        return self.__wob_f

    @wob_f.setter
    def wob_f(self, v):
        self.__wob_f = v

    @property
    def wob_b(self):
        return self.__wob_b

    @wob_b.setter
    def wob_b(self, v):
        self.__wob_b = v

    @property
    def mp_jm_d(self):
        return self.__mp_jm_d

    @mp_jm_d.setter
    def mp_jm_d(self, v):
        self.__mp_jm_d = v

    @property
    def mp_jl_d(self):
        return self.__mp_jl_d

    @mp_jl_d.setter
    def mp_jl_d(self, v):
        self.__mp_jl_d = v

    @property
    def mp_jm_t(self):
        return self.__mp_jm_t

    @mp_jm_t.setter
    def mp_jm_t(self, v):
        self.__mp_jm_t = v

    @property
    def mp_jl_t(self):
        return self.__mp_jl_t

    @mp_jl_t.setter
    def mp_jl_t(self, v):
        self.__mp_jl_t = v

    @property
    def rsbi(self):
        return self.__rsbi

    @rsbi.setter
    def rsbi(self, v):
        self.__rsbi = v


class Target0():
    def __init__(self):
        super().__init__()
        self.__rr = 0
        self.__v_t = 0
        self.__ve = 0
        self.__wob = 0
        self.__rsbi = 0
        self.__mp_jl_d = 0
        self.__mp_jm_d = 0
        self.__mp_jl_t = 0
        self.__mp_jm_t = 0

    @property
    def rr(self):
        return self.__rr

    @rr.setter
    def rr(self, v):
        self.__rr = v

    @property
    def v_t(self):
        return self.__v_t

    @v_t.setter
    def v_t(self, v):
        self.__v_t = v

    @property
    def ve(self):
        return self.__ve

    @ve.setter
    def ve(self, v):
        self.__ve = v

    @property
    def wob(self):
        return self.__wob

    @wob.setter
    def wob(self, v):
        self.__wob = v

    @property
    def rsbi(self):
        return self.__rsbi

    @rsbi.setter
    def rsbi(self, v):
        self.__rsbi = v

    @property
    def mp_jm_d(self):
        return self.__mp_jm_d

    @mp_jm_d.setter
    def mp_jm_d(self, v):
        self.__mp_jm_d = v

    @property
    def mp_jl_d(self):
        return self.__mp_jl_d

    @mp_jl_d.setter
    def mp_jl_d(self, v):
        self.__mp_jl_d = v

    @property
    def mp_jm_t(self):
        return self.__mp_jm_t

    @mp_jm_t.setter
    def mp_jm_t(self, v):
        self.__mp_jm_t = v

    @property
    def mp_jl_t(self):
        return self.__mp_jl_t

    @mp_jl_t.setter
    def mp_jl_t(self, v):
        self.__mp_jl_t = v
