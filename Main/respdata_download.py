import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, DLToLD, measure
from Classes.DataDownload import MYFTP, RecordDetect
from Classes.ORM.main import ZresParam, OutcomeExWean, ExtubePrep, WeanPrep, db, fn

main_mode = 'Wean'
mode_info = {
    'Extube': {
        'class': ExtubePrep,
        'tag': [3004, 129]
    },
    'Wean': {
        'class': WeanPrep,
        'tag': []
    }
}


@measure
def main() -> None:
    Ftp = __FtpGen()

    dst_class = mode_info[main_mode]['class']
    dst_flag = mode_info[main_mode]['tag']
    data_path = Path(ConfigRead('ServerData'))
    save_path = Path(ConfigRead('WaveData', main_mode))
    quer_list = RidQuery(main_mode)

    db.create_tables([dst_class])

    def __RecDownload(query_obj: any) -> dict:
        ins_d = dst_class().ObjToDict()
        main_p = RecordDetect(query_obj, ins_d, main_mode)
        main_p.InfoDetection(ZresParam, fn.MAX, dst_flag)
        main_p.RidDetection(Ftp.ftp, data_path)
        main_p.RecsDetection(Ftp.ftp, save_path)
        return main_p.dst

    def __RecRegister(ins_d: dict) -> None:
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


def RidQuery(q_mode: str, p_start: int = 0) -> list:
    src_0, src_1 = OutcomeExWean, ZresParam

    join_info = {
        'dest': src_0,
        'on': (src_0.pid == src_1.pid) & (src_1.rid != None),
        'attr': 'binfo'  # key value for query B columns from A  
    }

    # Mode Define

    if not type(q_mode) == str:
        print('Wrong Type Query!')
        return
    elif q_mode == 'Extube':
        end_t = src_0.ex_t
        end_i = src_0.ex_s
    elif q_mode == 'Wean':
        end_t = src_0.we_t
        end_i = src_0.we_s
    else:
        print('No such data collecting mode!')
        return

    # Query requirement

    pew_time = fn.date_trunc
    col_set = [src_1.pid, src_1.rid]
    col_que = [src_1.pid, src_1.rid, src_0.icu, end_t, end_i]
    cond_0 = end_t != None
    cond_1 = pew_time('day', src_1.rec_t) == pew_time('day', end_t)
    cond_2 = src_1.pid > p_start
    condition = cond_0 & cond_1 & cond_2

    query_list = src_1.select(*col_que).join(
        **join_info).where(condition).group_by(*col_set).order_by(*col_set)

    return query_list


if __name__ == '__main__':
    main()