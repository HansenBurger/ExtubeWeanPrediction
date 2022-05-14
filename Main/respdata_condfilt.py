import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

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