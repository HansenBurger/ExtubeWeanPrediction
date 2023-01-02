import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.ORM.basic import OutcomeExWean
from Classes.ORM.expr import PatientInfo
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12, WeanPSV, WeanSumP12


class Basic():
    def __init__(self) -> None:
        pass


class StaticData(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__p_basic_i = PatientInfo
        self.__op_basic_i = OutcomeExWean
        self.__cate_info = {
            'Extube': {
                'PSV': ExtubePSV,
                'SumP12': ExtubeSumP12,
                'multirid': [6828867, 6750903, 6447979, 3469338, 2417319]
            },
            'Wean': {
                'PSV': WeanPSV,
                'SumP12': WeanSumP12,
                'multirid': []
            },
            'Supp': {
                'All': ['Reached mv time', 'Single RID'],
                'Nad': ['Reached mv time', 'No Aged or illness', 'Single RID'],
                'AorD': ['Reached mv time', 'Aged or illness', 'Single RID']
            }
        }
        self.__ind_range = {
            'rr': [],
            'v_t_i': [100, 800],
            've': [],
            'rsbi': [0, 220],
            'mp_jb_d': [],
            'mp_jb_t': [],
            'mp_jm_d': [],
            'mp_jm_t': [],
            'mp_jl_d': [],
            'mp_jl_t': [0.7, 1.2]
        }
        self.__psv_vms = ['SPONT', 'CPAP', 'APNEA VENTILATION']
        self.__methods = ['TD', 'HRA', 'HRV', 'ENT', 'PRSA']

    @property
    def p_basic_i(self):
        return self.__p_basic_i

    @property
    def op_basic_i(self):
        return self.__op_basic_i

    @property
    def cate_info(self):
        return self.__cate_info

    @property
    def ind_range(self):
        return self.__ind_range

    @property
    def psv_vms(self):
        return self.__psv_vms

    @property
    def methods(self):
        return self.__methods


class DynamicData(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__mode_n = ''
        self.__data_loc = ''
        self.__s_f_fold = ''
        self.__s_g_fold = ''
        self.__pid_dr_s = {}

    @property
    def mode_n(self):
        return self.__mode_n

    @mode_n.setter
    def mode_n(self, v: str):
        self.__mode_n = v

    @property
    def data_loc(self):
        return self.__data_loc

    @data_loc.setter
    def data_loc(self, v: Path):
        self.__data_loc = v

    @property
    def s_f_fold(self):
        return self.__s_f_fold

    @s_f_fold.setter
    def s_f_fold(self, v: Path):
        self.__s_f_fold = v

    @property
    def s_g_fold(self):
        return self.__s_g_fold

    @s_g_fold.setter
    def s_g_fold(self, v: Path):
        self.__s_g_fold = v

    @property
    def pid_dr_s(self):
        return self.__pid_dr_s

    @pid_dr_s.setter
    def pid_dr_s(self, v: dict):
        self.__pid_dr_s = v
