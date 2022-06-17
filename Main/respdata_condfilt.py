import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.DataFiltering import TablePrepFilt, RecTransmit, DistGenerate
from Classes.Func.KitTools import measure, ConfigRead, SaveGen
from Classes.ORM.basic import OutcomeExWean, ExtubePrep, WeanPrep
from Classes.ORM.cate import db, ExtubePSV, ExtubeSumP10, ExtubeSumP12, ExtubeNotPSV, WeanPSV, WeanSumP10, WeanSumP12, WeanNotPSV

mode_s = ['Extube', 'Wean']
p_name = 'DataFilt'
form_save = SaveGen(Path(ConfigRead('ResultSave', 'Form')), p_name)
graph_save = SaveGen(Path(ConfigRead('ResultSave', 'Graph')), p_name)
mode_info = {
    'Extube': {
        'class': ExtubePrep,
        'mv_t': (24, 2160),
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
        'mv_t': (24, 2160),
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
def main(mode_: str) -> None:

    src_0 = OutcomeExWean
    src_1 = mode_info[mode_]['class']
    dst_d = mode_info[mode_]['dst_c']
    s_f_p = form_save / mode_
    s_f_p.mkdir(parents=True, exist_ok=True)
    s_g_p = graph_save / mode_
    s_g_p.mkdir(parents=True, exist_ok=True)

    dst_table = list(i.__name__ for i in dst_d.values())
    ext_table = db.get_tables()

    if dst_table < ext_table:
        db.drop_tables(dst_d.values())
    db.create_tables(dst_d.values())

    filter = TablePrepFilt(src_0, src_1, mode_info[mode_]['d_e_s'])
    que_val, p_val = filter.ValQueGen(s_f_p, mode_info[mode_]['mv_t'])

    pid_ld = [ParaDistGet(mode_, que_val.where(src_1.pid == i)) for i in p_val]

    process = DistGenerate(pid_ld)
    process.DistInfo('vmd', s_f_p, s_g_p)
    process.DistInfo('spd', s_f_p, s_g_p)

    TableDistGet(mode_, s_f_p)


def ParaDistGet(mode_: str, que_o: any) -> dict:
    dst_c_d = mode_info[mode_]['dst_c']
    classifier = RecTransmit(que_o, 1800)
    classifier.PDistBuilt()
    classifier.PSVInsert(dst_c_d['PSV'], psv_vm)
    classifier.NotPSVInsert(dst_c_d['noPSV'], psv_vm)
    classifier.PSVSumPInsert(dst_c_d['Sump12'], psv_vm, 12)
    classifier.PSVSumPInsert(dst_c_d['Sump10'], psv_vm, 10)

    return classifier.p_d


def TableDistGet(mode_: str, save_path: Path) -> None:
    src_l = list(mode_info[mode_]['dst_c'].values())
    table_info = save_path / ('ProcessInfo.txt')
    with open(table_info, 'w') as f:
        for src in src_l:
            f.write(str(src.__name__ + ':\n'))
            col_group = [src.pid]
            cond_succ = src.e_s.contains('成功')
            cond_fail = src.e_s.contains('失败')
            tot_len = len(src.select().group_by(*col_group))
            succ_len = len(src.select().where(cond_succ).group_by(*col_group))
            fail_len = len(src.select().where(cond_fail).group_by(*col_group))
            f.write('\ttot: {0}, succ: {1} | fail: {2}\n'.format(
                tot_len, succ_len, fail_len))


if __name__ == '__main__':
    for mode_ in mode_s:
        main(mode_)