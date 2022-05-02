import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead, measure
from Classes.DataDownload import MYFTP, RecordDetect
from Classes.ORM.main import ZresParam, OutcomeExWean, ExtubePrep, WeanPrep, db, fn

main_mode = 'Extube'


def main() -> None:
    Ftp = __FtpGen()
    data_path = Path(ConfigRead('ServerData'))
    save_path = Path(ConfigRead(main_mode))
    quer_list = RidQuery(main_mode)

    dst_class = {'Extube': ExtubePrep, 'Wean': WeanPrep}
    dst_flag = {'Extube': [3004, 129], 'Wean': []}

    try:
        db.create_tables([dst_class[main_mode]])
    except:
        print('Table already exist!')

    @measure
    def __RecDownload(query_obj: any) -> dict:
        ins_d: dict = dst_class().ObjToDict()
        main_p = RecordDetect(query_obj, ins_d)
        main_p.InfoDetection(ZresParam, fn.MAX, dst_flag[main_mode])
        main_p.RidDetection(Ftp.ftp, data_path)
        main_p.RecsDetection(Ftp.ftp, save_path)
        return main_p.dst

    def __RecRegister(src_dict: dict) -> None:
        DL = {k: [v] if type(v) != list else v for k, v in src_dict.items()}
        max_len = max(len(DL.values()))
        DL = {
            k: v * max_len if len(v) < max_len else v
            for k, v in src_dict.items()
        }
        LD = [dict(zip(DL, t)) for t in zip(*DL.values())]
        dst_class.insert_many(LD).execute()

    Ftp.FtpLogin()

    for que_o in quer_list:
        ins_d: dict = dst_class().ObjToDict()
        main_p = RecordDetect(que_o, ins_d)
        main_p.InfoDetection(ZresParam, fn.MAX, dst_flag[main_mode])
        main_p.RidDetection(Ftp.ftp, data_path)
        main_p.RecsDetection(Ftp.ftp, save_path)
        dst_d = main_p.dst


def __FtpGen() -> MYFTP:
    '''
    Generate server cursor by server info
    '''
    ftp = MYFTP(**ConfigRead('Server'))
    return ftp


def RidQuery(q_mode: str) -> list:
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
    elif q_mode == 'Wean':
        end_t = src_0.we_t
    else:
        print('No such data collecting mode!')
        return

    # Query requirement

    pew_time = fn.date_trunc
    col_set = [src_1.pid, src_1.rid]
    col_que = [src_1.pid, src_1.rid, src_0.icu, end_t]
    cond_0 = end_t != None
    cond_1 = pew_time('day', src_1.rec_t) == pew_time('day', end_t)
    condition = cond_0 & cond_1

    query_list = src_1.select(*col_que).join(
        **join_info).where(condition).group_by(*col_set).order_by(*col_set)

    return query_list


if __name__ == '__main__':
    main()