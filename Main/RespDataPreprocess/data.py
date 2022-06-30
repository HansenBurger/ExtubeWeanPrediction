import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.ORM.basic import ExtubePrep, WeanPrep, OutcomeExWean
from Classes.ORM.cate import WeanPSV, WeanSumP10, WeanSumP12, WeanNotPSV
from Classes.ORM.cate import ExtubePSV, ExtubeSumP10, ExtubeSumP12, ExtubeNotPSV


class Basic():
    def __init__(self) -> None:
        pass


class StaticData(Basic):
    def __init__(self) -> None:
        super().__init__()
        # mv still time set (days)
        self.__mv_range = (48, 2160)

        # mode info
        self.__mode_info = {
            'Extube': {
                'class': ExtubePrep,  # extube class
                'tag': [3004, 129],  # operation tags
                'mv_t':
                self.__mv_range,  # machine vent still time (follow mv_range)
                'd_e_s': OutcomeExWean.ex_s,  # extube statu in table
                'd_e_t': OutcomeExWean.ex_t,  # extube time in table
                'dst_c': {
                    'PSV': ExtubePSV,  # psv table class
                    'noPSV': ExtubeNotPSV,  # not psv table class
                    'Sump10': ExtubeSumP10,  # psv & sump-10 table class
                    'Sump12': ExtubeSumP12  # psv & sump-12 table class
                }
            },
            'Wean': {
                'class': WeanPrep,  # wean class
                'tag': [],
                'mv_t':
                self.__mv_range,  # machine vent still time (follow mv_range)
                'd_e_s': OutcomeExWean.we_s,  # wean statu in table
                'd_e_t': OutcomeExWean.we_t,  # wean time in table
                'dst_c': {
                    'PSV': WeanPSV,  # psv table class
                    'noPSV': WeanNotPSV,  # not psv table class
                    'Sump10': WeanSumP10,  # psv & sump-10 table class
                    'Sump12': WeanSumP12  # psv & sump-12 table class
                }
            }
        }

        # vefication setting
        self.__verify_st = {
            'para_t_st': [10, 910, 1810, 2710],  # para time st (second)
            'para_n_st': ['st_mode', 'st_peep', 'st_ps']  # para name st
        }

        self.__condfilt_st = {
            'para_t_scale': 1800,  # paras observation scale
            'psv_vm_names': ['SPONT', 'CPAP', 'APNEA VENTILATION']
        }

    @property
    def mode_info(self):
        return self.__mode_info

    @property
    def verify_st(self):
        return self.__verify_st

    @property
    def condfilt_st(self):
        return self.__condfilt_st


class DynamicData(Basic):
    def __init__(self) -> None:
        super().__init__()