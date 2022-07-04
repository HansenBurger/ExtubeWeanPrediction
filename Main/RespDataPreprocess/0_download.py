import sys
from pathlib import Path
from datetime import datetime
from data import StaticData
from funcs.data_detection import MYFTP, RecordDetect

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, DLToLD, measure
from Classes.ORM.basic import ZresParam, OutcomeExWean, db, fn

mode_info = StaticData().mode_info


@measure
def main(mode_: str) -> None:
    Ftp = __FtpGen()

    mode_set = mode_info[mode_]
    dst_class = mode_set['class']
    dst_flag = mode_set['tag']
    data_path = Path(ConfigRead('ServerData'))
    save_path = Path(ConfigRead('WaveData', mode_))
    quer_list = RidQuery(mode_, mv_still=mode_set['mv_t'])

    db.create_tables([dst_class])

    def __RecDownload(query_obj: any) -> dict:
        '''
        Download records in selected rid
        '''
        ins_d = dst_class().ObjToDict()
        main_p = RecordDetect(query_obj, ins_d, mode_)
        main_p.InfoDetection(ZresParam, fn.MAX, dst_flag)
        main_p.RidDetection(Ftp.ftp, data_path)
        main_p.RecsDetection(Ftp.ftp, save_path)
        return main_p.dst

    def __RecRegister(ins_d: dict) -> None:
        '''
        Update records info into table
        '''
        ins_l = DLToLD(ins_d)
        dst_class.insert_many(ins_l).on_conflict('replace').execute()

    Ftp.FtpLogin()

    for que_o in quer_list:
        t_s = datetime.now()
        ins_d = __RecDownload(que_o)
        __RecRegister(ins_d)
        t_e = datetime.now()

        print('Pid: {0}, Rid: {1}, ValRec: {2}, Process Time: {3}'.format(
            ins_d['pid'], ins_d['rid'],
            len(ins_d['rec_t']) if ins_d['rec_t'] else 0, t_e - t_s))

    Ftp.FtpLogout()


def __FtpGen() -> MYFTP:
    '''
    Generate server cursor by server info
    '''
    ftp = MYFTP(**ConfigRead('Server'))
    return ftp


def RidQuery(q_mode: str,
             mv_still: tuple = (48, 2160),
             p_range: range = None) -> list:
    '''
    Query the valid Rids in the patients' op-day
    q_mode: query mode
    mv_still: machine vent still time as part of filter
    p_range: patient range
    return: rid list in op day
    '''
    src_0, src_1 = OutcomeExWean, ZresParam

    join_info = {
        'dest': src_0,
        'on': (src_0.pid == src_1.pid) & (src_1.rid != None),
        'attr': 'binfo'  # key value for query B columns from A  
    }

    end_i = mode_info[q_mode]['d_e_s']
    end_t = mode_info[q_mode]['d_e_t']

    # Query requirement

    pew_time = fn.date_trunc
    col_set = [src_1.pid, src_1.rid]
    col_que = [src_1.pid, src_1.rid, src_0.icu, end_t, end_i]
    c_op_exist = end_t != None
    c_same_day = pew_time('day', src_1.rec_t) == pew_time('day', end_t)
    c_mv_still = (src_0.mv_t >= mv_still[0]) & (src_0.mv_t <= mv_still[1])

    if not p_range:
        c_pid = src_1.pid > 0
    else:
        c_pid = (src_1.pid >= p_range[0]) & (src_1.pid <= p_range[-1])

    condition = c_op_exist & c_same_day & c_mv_still & c_pid

    query_list = src_1.select(*col_que).join(
        **join_info).where(condition).group_by(*col_set).order_by(*col_set)

    return query_list


if __name__ == '__main__':
    main(sys.argv[1])