import sys
import pandas as pd
from ftplib import FTP
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.ORM.basic import ZresParam, fn
from Classes.Func.KitTools import PathVerify, ConfigRead, DLToLD, SaveGen
from Classes.Func.DiagramsGen import PlotMain
from Main.RespDataPreprocess.funcs import data_detection

pids = [6913426, 3837466, 6640624, 6003532]
s_main_p = SaveGen(Path(ConfigRead('ResultSave', 'Mix')), 'pcv_var')
s_form_p, s_graph_p = s_main_p / 'form', s_main_p / 'graph'
'''
Process workflow
1. Copy data from server
    1.1 Source: zres_param, pid_required
    1.2 Condition: Unique Rid, total vent time ≤ 2 days (manually)
    1.3 Result: table like Extube_PSV
2. Calculate the breathing variability in total vent time
    2.1 Condition: mode -- PCV
    2.2 Calculation: 
        (1) Values of respiratory parameters throughout ventilation
        (2) Variability results per hour, per 30 min?(changable)
    2.3 Storage Intermediate:
3. Display results
    3.1 resp param trend plot(line plot total, scatter plot per hour)
    3.2 variability trend plot
'''


def RidQuery(que_src: any = ZresParam, pid_in_s: list = pids) -> list:

    que_tot = []

    for pid in pid_in_s:
        gp_set = que_src.rid
        col_que = [que_src.pid, que_src.rid, que_src.icu]
        col_set = [que_src.pid, que_src.rec_t]
        cond_0 = que_src.pid == pid
        cond_1 = (que_src.rid != None) & (que_src.icu != None)
        que_pid = que_src.select(*col_que).where(cond_0
                                                 & cond_1).order_by(*col_set)

        if len(que_pid.group_by(gp_set)) < 1:
            continue
        elif len(que_pid.group_by(gp_set)) > 1:
            continue
        else:
            que_tot.append(que_pid[0])

    return que_tot


def __FtpGen() -> data_detection.MYFTP:
    '''
    Generate server cursor by server info
    '''
    ftp = data_detection.MYFTP(**ConfigRead('Server'))
    return ftp


class RecordDetect(data_detection.RecordDetect):
    def __init__(self, obj_src: any, dic_dst: dict, main_mode: str) -> None:
        super().__init__(obj_src, dic_dst, main_mode)
        self.__src = obj_src
        self.__dst = dic_dst

    @property
    def dst(self):
        return self.__dst

    def InfoDetection_n(self, table: any, func: any) -> None:

        cond_0 = table.rid == self.__src.rid
        end_t = table.select(func(table.rec_t)).where(cond_0).scalar()

        self.__dst['pid'] = self.__src.pid
        self.__dst['rid'] = self.__src.rid
        self.__dst['icu'] = self.__src.icu
        self.__dst['tail_t'] = end_t


def main(mode_: str = 'PCV'):
    Ftp = __FtpGen()

    data_src_p = Path(ConfigRead('ServerData'))
    data_dst_p = Path(ConfigRead('WaveData', mode_))

    Ftp.FtpLogin()

    que_l = RidQuery()
    que_dst = []
    for que_i in que_l:
        que_p = RecordDetect(que_i, {}, mode_)
        que_p.InfoDetection_n(ZresParam, fn.MAX)
        que_p.RidDetection(Ftp.ftp, data_src_p)
        que_p.RecsDetection(Ftp.ftp, data_dst_p, way_st='')
        que_dst_ld = DLToLD(que_p.dst)
        que_dst.extend(que_dst_ld)
    que_df = pd.DataFrame(que_dst)

    que_df.to_csv(s_main_p / 'pcv.csv', index=False)

    Ftp.FtpLogout()


if __name__ == "__main__":
    main()