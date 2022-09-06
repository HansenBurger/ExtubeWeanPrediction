class func():
    def __init__(self) -> None:
        pass


class Resp(func):
    def __init__(self):
        super().__init__()
        self.__wid = None
        self.__val = None
        self.__t_i = 0.0
        self.__t_e = 0.0
        self.__i_e = 0.0
        self.__pip = 0.0
        self.__peep = 0.0
        self.__mpaw = 0.0
        self.__rr = 0.0
        self.__v_t_i = 0.0
        self.__v_t_e = 0.0
        self.__ve = 0.0
        self.__rsbi = 0.0
        self.__mp_jb_a = 0.0
        self.__mp_jb_b = 0.0
        self.__mp_jb_t = 0.0
        self.__mp_jb_d = 0.0
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
    def t_i(self):
        return self.__t_i

    @t_i.setter
    def t_i(self, v):
        self.__t_i = v

    @property
    def t_e(self):
        return self.__t_e

    @t_e.setter
    def t_e(self, v):
        self.__t_e = v

    @property
    def i_e(self):
        return self.__i_e

    @i_e.setter
    def i_e(self, v):
        self.__i_e = v

    @property
    def pip(self):
        return self.__pip

    @pip.setter
    def pip(self, v):
        self.__pip = v

    @property
    def peep(self):
        return self.__peep

    @peep.setter
    def peep(self, v):
        self.__peep = v

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
    def mp_jb_d(self):
        return self.__mp_jb_d

    @mp_jb_d.setter
    def mp_jb_d(self, v):
        self.__mp_jb_d = v

    @property
    def mp_jb_a(self):
        return self.__mp_jb_a

    @mp_jb_a.setter
    def mp_jb_a(self, v):
        self.__mp_jb_a = v

    @property
    def mp_jb_t(self):
        return self.__mp_jb_t

    @mp_jb_t.setter
    def mp_jb_t(self, v):
        self.__mp_jb_t = v

    @property
    def mp_jb_b(self):
        return self.__mp_jb_b

    @mp_jb_b.setter
    def mp_jb_b(self, v):
        self.__mp_jb_b = v

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
        self.__t_i = 0  # Inspiratory Time
        self.__t_e = 0  # Expiratory Time
        self.__t_tot = 0  # Respiratory Time
        self.__i_e = 0  # Inspiratory to Expiratory Ratio
        self.__pip = 0  # Peak Inspiratory Pressure
        self.__peep = 0  # Positive End-Expiratory Pressure
        self.__rr = 0  # Respiratory Rate
        self.__v_t = 0  # Tidal Volume
        self.__ve = 0  # Minute Ventilation
        self.__rsbi = 0  # Rapid Shallow Breathing Index
        self.__mp_jb_d = 0  # Dynamic Mechanical Power (J/Breath)
        self.__mp_jb_t = 0  # Total Mechanical Power (J/Breath)
        self.__mp_jl_d = 0  # Dynamic Mechanical Power (J/L)
        self.__mp_jm_d = 0  # Dynamic Mechanical Power (J/min)
        self.__mp_jl_t = 0  # Total Mechanical Power (J/L)
        self.__mp_jm_t = 0  # Total Mechanical Power (J/min)

    @property
    def t_i(self):
        '''
        Inspiratory Time (T_e)
        '''
        return self.__t_i

    @t_i.setter
    def t_i(self, v):
        '''
        :param v: T_i (float, unit: s)
        '''
        self.__t_i = v

    @property
    def t_e(self):
        '''
        Expiratory Time (T_e)
        '''
        return self.__t_e

    @t_e.setter
    def t_e(self, v):
        '''
        :param v: T_e (float, unit: s)
        '''
        self.__t_e = v

    @property
    def t_tot(self):
        '''
        Respiratory Time (T_tot)
        '''
        return self.__t_tot

    @t_tot.setter
    def t_tot(self, v):
        '''
        :param v: T_tot (float, unit: s)
        '''
        self.__t_tot = v

    @property
    def i_e(self):
        '''
        Inspiratory to Expiratory Ratio (I:E Ratio)
        '''
        return self.__i_e

    @i_e.setter
    def i_e(self, v):
        '''
        :param v: I:E Ratio (float, unit: x)
        '''
        self.__i_e = v

    @property
    def pip(self):
        '''
        Peak Inspiratory Pressure (PIP)
        '''
        return self.__pip

    @pip.setter
    def pip(self, v):
        '''
        :param v: PIP (float, unit: cmH2O)
        '''
        self.__pip = v

    @property
    def peep(self):
        '''
        Positive End-Expiratory Pressure (PEEP)
        '''
        return self.__peep

    @peep.setter
    def peep(self, v):
        '''
        :param v: PEEP (float, unit: cmH2O)
        '''
        self.__peep = v

    @property
    def rr(self):
        '''
        Respiratory Rate (RR)
        '''
        return self.__rr

    @rr.setter
    def rr(self, v):
        '''
        :param v: RR (float, unit: N/min)
        '''
        self.__rr = v

    @property
    def v_t(self):
        '''
        Tidal Volume (V_T)
        '''
        return self.__v_t

    @v_t.setter
    def v_t(self, v):
        '''
        :param v: V_T (float, unit: ml)
        '''
        self.__v_t = v

    @property
    def ve(self):
        '''
        Minute Ventilation (MV)
        '''
        return self.__ve

    @ve.setter
    def ve(self, v):
        '''
        :param v: MV (float, unit: L/min)
        '''
        self.__ve = v

    @property
    def mp_jb_d(self):
        '''
        Dynamic Mechanical Power (J/breath)
        '''
        return self.__mp_jb_d

    @mp_jb_d.setter
    def mp_jb_d(self, v):
        '''
        :param v: MP_d(J/breath) (float, unit: J/L)
        '''
        self.__mp_jb_d = v

    @property
    def mp_jb_t(self):
        '''
        Dynamic Mechanical Power (J/breath)
        '''
        return self.__mp_jb_t

    @mp_jb_t.setter
    def mp_jb_t(self, v):
        '''
        :param v: MP_d(J/breath) (float, unit: J/L)
        '''
        self.__mp_jb_t = v

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
