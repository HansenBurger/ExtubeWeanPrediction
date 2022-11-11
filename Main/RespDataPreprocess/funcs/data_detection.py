import sys
import pandas as pd
from ftplib import FTP
from shutil import copy
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path.cwd()))

from WaveDataProcess import BinImport


class Basic():
    def __init__(self) -> None:
        self.__format_l = ['%Y-%m-%d %H:%M:%S']
        pass

    def __STtrans(self, time_string: str) -> datetime:
        '''
        Convert the time string to datetime on cond
        time_string: the time string prepared tobe converted
        '''
        time_time = None
        for t_format in self.__format_l:
            try:
                time_time = datetime.strptime(time_string, t_format)
            except:
                continue
        if not time_time:
            print('No Match format !')
        return time_time

    def __TPtrans(self, time_time: datetime) -> str:
        '''
        Convert datetime to part of file path
        time_time: record's datetime info
        '''
        time_year = str(time_time.year)
        time_month = str(time_time.month).rjust(2, '0')
        time_path = time_year + time_month
        return time_path

    def __FileCheck(self, path: Path, ftp: FTP):
        '''
        Check if the file exist in the path
        path: purepath for detection
        ftp: FTP server cursor
        '''
        try:
            ftp.cwd(str(path.parents[0]))
            file_check = lambda f: f in ftp.nlst()
            check = file_check(path.parts[-1])
            return check
        except:
            return False

    def __FileDownload(self,
                       src_p: str,
                       dst_p: str,
                       ftp: FTP,
                       force: bool = True):
        '''
        Download File from source to destination
        src_p: source path
        dst_p: destination path
        ftp: FTP: server cursor
        '''
        src_p = str(src_p) if type(src_p) != str else src_p
        dst_p = str(dst_p) if type(dst_p) != str else dst_p

        if force:
            pass
        else:
            if not dst_p.is_file():
                pass
            else:
                print('File Already exist')
                return

        with open(dst_p, 'wb') as download_file:
            ftp.retrbinary('RETR {0}'.format(src_p), download_file.write)


class MYFTP(Basic):
    def __init__(self,
                 host_s: str,
                 port_n: int,
                 user: str = '',
                 passwd: str = ''):
        super().__init__()
        self.__usr = user
        self.__pwd = passwd
        self.__ftp = FTP()
        self.__ftp.connect(host=host_s, port=port_n)

    @property
    def ftp(self):
        return self.__ftp

    def FtpLogin(self):
        '''
        The ftp obj can only be used after log in
        '''
        self.__ftp.login(user=self.__usr, passwd=self.__pwd)
        print(self.__ftp.getwelcome())

    def FtpLogout(self):
        '''
        Have to new one ftp obj for processing after log out
        '''
        self.__ftp.quit()


class RecordDetect(Basic):
    def __init__(self, obj_src: any, dic_dst: dict, main_mode: str) -> None:
        super().__init__()
        self.__mode = main_mode
        self.__src = obj_src
        self.__dst = dic_dst.copy()
        self.__bicheck = lambda x, y, z, func: func(x, z) & func(
            y, z)  # check the folder and zif
        self.__icu_list = [
            'CCU', 'EICU', 'ICU3F', 'ICU4F', 'xsICU3F', 'xsICU4F'
        ]

    @property
    def dst(self):
        return self.__dst

    def __TimeRange(self, t_jug: datetime) -> list:
        '''
        Determine if the input time meets the demand
        t_jug: time judge(get the range around the time judge)
        '''
        t_jug = self._Basic__TFtrans(t_jug) if type(t_jug) == str else t_jug

        t_sep_early = {'hours': 3, 'minutes': 10}
        t_sep_delay = {'hours': 6, 'minutes': 5}

        t_pre = t_jug - timedelta(**t_sep_early)
        t_post = t_jug + timedelta(**t_sep_delay)

        return t_pre, t_post

    def InfoDetection(self, table: any, func: any, flag: list = []) -> None:
        '''
        Rid basic info detection
        table: zres_param
        func: function certain the maxium value in table
        flag: flag for whether or not to perform the operation
        '''

        cond_0 = table.rid == self.__src.rid
        end_t = table.select(func(table.rec_t)).where(cond_0).scalar()

        if not flag:
            op_exist = False
        else:
            cond_1 = table.pid == self.__src.pid
            cond_2 = table.rec_i == flag[0]
            cond_3 = table.rec_f == flag[1]
            op_exist = table.select().where(cond_1 & cond_2 & cond_3).exists()

        self.__dst['pid'] = self.__src.pid
        self.__dst['rid'] = self.__src.rid
        self.__dst['icu'] = self.__src.binfo.icu
        self.__dst['tail_t'] = end_t
        self.__dst['opt'] = op_exist

        if self.__mode == 'Extube':
            self.__dst['e_t'] = self.__src.binfo.ex_t
            self.__dst['e_s'] = self.__src.binfo.ex_s
        elif self.__mode == 'Wean':
            self.__dst['e_t'] = self.__src.binfo.we_t
            self.__dst['e_s'] = self.__src.binfo.we_s

    def RidDetection(self, ftp: FTP, main_loc: Path) -> None:
        '''
        Rid route detection
        ftp: server cursor
        main_loc: resp data loc
        '''
        icu = self.__dst['icu']
        tin = self._Basic__TPtrans(self.__dst['tail_t'])
        rid = self.__dst['rid']

        def ICUDefine(icu):
            p_folder = main_loc / icu / tin / rid
            p_file = main_loc / icu / tin / (rid + '.zif')

            p_status = self.__bicheck(p_folder, p_file, ftp,
                                      self._Basic__FileCheck)
            return p_status

        if ICUDefine(icu):
            pass
        else:
            icu = ''  # redirect the file path(icu)
            for icu_t in self.__icu_list:
                if not ICUDefine(icu_t):
                    continue
                else:
                    icu = icu_t
                    break

        de_p = main_loc / icu / tin / rid if icu else None
        self.__dst['rot'] = str(de_p) if de_p else None  # Table insert support

    def __ZifDataFilt(self,
                      zif_file: str,
                      way_st: str = 'e_t') -> pd.DataFrame:
        df_rec = pd.DataFrame(BinImport.RidData(zif_file).RecordListGet())

        if df_rec.empty:
            pass
        else:
            df_rec = df_rec.loc[df_rec.s_t.apply(type) == pd.Timestamp]
            df_rec = df_rec.sort_values(by='s_t')

            if not way_st == 'e_t':
                pass
            else:
                ext_pre, ext_post = self.__TimeRange(self.__dst[way_st])
                filt_0 = df_rec.s_t > ext_pre
                filt_1 = df_rec.s_t < ext_post
                df_rec = df_rec[filt_0 & filt_1]

        return df_rec

    def RecsDetection(self, ftp: FTP, save_loc: Path, **kwargs):
        '''
        Collect all records meeting requirement in RID folder
        ftp: server cursor
        save_loc: local path to maintain the data
        '''

        if not self.__dst['rot']:
            return

        rot = Path(self.__dst['rot'])
        rid = self.__dst['rid']
        zif_file = rid + '.zif'

        self._Basic__FileDownload(str(rot) + '.zif', zif_file, ftp)
        df_rec = self.__ZifDataFilt(zif_file, **kwargs)

        self.__dst['zdt'] = None if df_rec.empty else []
        self.__dst['zpx'] = None if df_rec.empty else []
        self.__dst['rec_t'] = None if df_rec.empty else []

        if df_rec.empty:
            pass
        else:
            df_rec = df_rec.reset_index(drop=True)
            for i in df_rec.index:
                row = df_rec.iloc[i]
                src_zdt = rot / (row.fid + '.zdt')
                src_zpx = rot / (row.fid + '.zpx')

                zdt_sta = row.fid if self._Basic__FileCheck(src_zdt,
                                                            ftp) else None
                zpx_sta = row.fid if self._Basic__FileCheck(src_zpx,
                                                            ftp) else None

                if not zdt_sta or not zpx_sta:
                    pass
                else:
                    tin_true = self._Basic__TPtrans(row.s_t)
                    dst_loc = save_loc / tin_true / rid
                    dst_loc.mkdir(parents=True, exist_ok=True)
                    dst_zdt = dst_loc / (row.fid + '.zdt')
                    dst_zpx = dst_loc / (row.fid + '.zpx')

                    self._Basic__FileDownload(src_zdt, dst_zdt, ftp)
                    self._Basic__FileDownload(src_zpx, dst_zpx, ftp)

                    copy(str(Path.cwd() / zif_file), str(dst_loc / zif_file))

                # have to change rec_t from 'timestamp' to 'datetime'
                self.__dst['rec_t'].append(row.s_t.to_pydatetime())
                self.__dst['zdt'].append(zdt_sta)
                self.__dst['zpx'].append(zpx_sta)

        Path(Path.cwd() / zif_file).unlink()