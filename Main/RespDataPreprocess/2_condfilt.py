import sys
from pathlib import Path
from data import StaticData
from funcs.data_classifications import TablePrepFilt, RecTransmit, DistGenerate

sys.path.append(str(Path.cwd()))

from Classes.ORM.cate import db
from Classes.ORM.basic import OutcomeExWean
from Classes.Func.KitTools import measure, ConfigRead, SaveGen

p_name = 'DataFilt'
static = StaticData()
mode_info = static.mode_info
psv_vm = static.condfilt_st['psv_vm_names']
ob_scale = static.condfilt_st['para_t_scale']


@measure
def main(mode_: str) -> None:

    src_0 = OutcomeExWean
    src_1 = mode_info[mode_]['class']
    dst_d = mode_info[mode_]['dst_c']
    s_p = SaveGen(Path(ConfigRead('ResultSave', 'Mix')),
                  '_'.join([p_name, mode_]))
    s_f_p = s_p / 'Form'
    s_f_p.mkdir(parents=True, exist_ok=True)
    s_g_p = s_p / 'Chart'
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
    classifier = RecTransmit(que_o, ob_scale)
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
    main(sys.argv[1])