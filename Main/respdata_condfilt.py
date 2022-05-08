#TODO following
'''
Three things
1. save basic filting result(which include):
    BY MODE = [Extube, Wean]
    --- respdata query ----
    1. Total p, Exist p (Succ | Fail)
    2. valid p (Succ | Fail)
    3. invalid p (Succ | Fail)
        1. p cant find rid (no rid in op-day | rid no route)
        2. p cant find any rec in rid (zdt or zpx not exist)
    ---- respdata verify ----
    1. valid p (Succ | Fail)
    2. invalid p (Succ | Fail)
        1. wave info not exist (may not use)
        2. machine name is '840-4' or ''840-22'
        3. vent time less than 600s (may not use)

2. query data
    1. generate pid gp
    2. extube/wean time re-ensure
        1. seperate dataframe by op_end time
        2. check if psv mode in df_down(change time set)
        3.  
'''
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.DiagramsGen import PlotMain
from Classes.DataFiltering import TablePrepFilt, RecTransmit, DistGenerate
from Classes.Func.KitTools import measure, ConfigRead, SaveGen
from Classes.ORM.basic import OutcomeExWean, ExtubePrep, WeanPrep
from Classes.ORM.cate import db, ExtubePSV, ExtubeSumP10, ExtubeSumP12, ExtubeNotPSV, WeanPSV, WeanSumP10, WeanSumP12, WeanNotPSV

mode_ = 'Wean'
save_name = mode_ + '_datafilt'
mode_info = {
    'Extube': {
        'class': ExtubePrep,
        'd_e_s': OutcomeExWean.ex_s,
        'dst_c': {
            'PSV': ExtubePSV,
            'noPSV': ExtubeNotPSV,
            'Sump10': ExtubeSumP10,
            'Sump12': ExtubeSumP12
        }
    },
    'Wean': {
        'class': WeanPrep,
        'd_e_s': OutcomeExWean.we_s,
        'dst_c': {
            'PSV': WeanPSV,
            'noPSV': WeanNotPSV,
            'Sump10': WeanSumP10,
            'Sump12': WeanSumP12
        }
    }
}
psv_vm = ['SPONT', 'CPAP', 'APNEA VENTILATION']


@measure
def main() -> None:

    src_0 = OutcomeExWean
    src_1 = mode_info[mode_]['class']
    dst_d = mode_info[mode_]['dst_c']
    s_f_p = SaveGen(Path(ConfigRead('ResultSave', 'Form')), save_name)
    s_g_p = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), save_name)

    if list(dst_d.keys()) < db.get_tables():
        db.drop_tables(dst_d.values())
    db.create_tables(dst_d.values())

    filter = TablePrepFilt(src_0, src_1, mode_info[mode_]['d_e_s'])
    que_val, p_val = filter.ValQueGen(s_f_p)

    pid_ld = [ParaDistGet(que_val.where(src_1.pid == i)) for i in p_val]

    process = DistGenerate(pid_ld)
    process.DistInfo('vmd', s_f_p, s_g_p)
    process.DistInfo('spd', s_f_p, s_g_p)


@measure
def ParaDistGet(que_o: any) -> dict:
    dst_c_d = mode_info[mode_]['dst_c']
    classifier = RecTransmit(que_o, 1800)
    classifier.PDistBuilt()
    classifier.PSVInsert(dst_c_d['PSV'], psv_vm)
    classifier.NotPSVInsert(dst_c_d['noPSV'], psv_vm)
    classifier.PSVSumPInsert(dst_c_d['Sump12'], psv_vm, 12)
    classifier.PSVSumPInsert(dst_c_d['Sump10'], psv_vm, 10)

    return classifier.p_d


if __name__ == '__main__':
    main()