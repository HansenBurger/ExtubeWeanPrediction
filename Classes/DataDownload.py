import re
import sys
from os import remove
from pathlib import Path
from ftplib import FTP
from shutil import copy
from datetime import datetime, timedelta

sys.path.append(str(Path.cwd()))
from WaveDataProcess import BinImport


class Basic():
    def __init__(self) -> None:
        self.__t_format = '%Y-%m-%d %H:%M:%S'
        pass

    def __TFtrans(self, t_str):
        t_out = datetime.strptime(t_str, self.__t_format)
        return t_out

    def __TPtrans(self, t_in):
        path = str(t_in.year) + str(t_in.month).rjust(2, '0')
        return path

    def __FileCheck(self, path, ftp):
        try:
            ftp.cwd(str(path.parents[0]))
            file_check = lambda f: f in ftp.nlst()
            check = file_check(path.parts[-1])
            return check
        except:
            return False

    def __FileDownload(self, src_p, dst_p, ftp):
        src_p = str(src_p) if type(src_p) != str else src_p
        dst_p = str(dst_p) if type(dst_p) != str else dst_p

        with open(dst_p, 'wb') as download_file:
            ftp.retrbinary('RETR {0}'.format(src_p), download_file.write)


class MYFTP(Basic):
    def __init__(self, host_s, port_n, user=None, passwd=None):
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


class RecordIdDetect(Basic):
    def __init__(self, obj_src, dic_dst):
        super().__init__()
        self.__src = obj_src
        self.__dst = dic_dst
        self.__bicheck = lambda x, y, z, func: func(x, z) & func(y, z)
        self.__icu_list = [
            'CCU', 'EICU', 'ICU3F', 'ICU4F', 'xsICU3F', 'xsICU4F'
        ]

    def InfoDetection(self, table, func):
        end_t = table.select(func(
            table.rec_t)).where(table.rid == self.__src.rid).scalar()

        self.__dst['pid'] = self.__src.pid
        self.__dst['rid'] = self.__src.rid
        self.__dst['end_t'] = end_t

    def RouteDetection(self, ftp, main_loc):
        icu = self.__src.binfo.icu
        tin = self.TPtrans(self.__dst['end_t'])
        rid = self.__dst['rid']

        or_p = main_loc / icu / tin / rid
        or_p_ = main_loc / icu / tin / (rid + '.zif')

        if self.__bicheck(or_p, or_p_, ftp, self.FileCheck):
            pass
        else:
            icu = None
            for icu_t in self.__icu_list:

                re_p = main_loc / icu_t / tin / rid
                re_p_ = main_loc / icu_t / tin / (rid + '.zif')

                if self.__bicheck(re_p, re_p_, ftp, self.FileCheck):
                    icu = icu_t
                    break
                else:
                    continue

        de_p = main_loc / icu / tin / rid if icu else None
        self.__dst['rot'] = de_p


class RecordNDetect(Basic):
    def __init__(self, obj_src, dic_dst) -> None:
        super().__init__()
        self.__src = obj_src
        self.__dst = dic_dst
        self.__tmp = obj_src.rid + '.zif'
        self.__patt = r'([A-Z]\w*_\d*)(202\d{1}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}){'

    def __ZifRead(self, src, ftp):
        self._FileDownload(str(src) + '.zif', self.__tmp, ftp)
        with open(self.__tmp, 'rb') as binary_file:
            list_ = binary_file.readlines()
        lines = [str(x) for x in list_]
        return lines

    def __TimeCheck(self, t_jug, t_juged):
        t_jug = self._Basic__TFtrans(t_jug) if type(t_jug) == str else t_jug
        t_juged = self._Basic__TFtrans(t_juged) if type(t_juged) == str else t_juged
        early_t = timedelta(hours=3, minutes=10)
        delay_t = timedelta(hours=6, minutes=5)
        t_pre, t_post = (t_jug - early_t), (t_jug + delay_t)
        result = False if t_juged > t_post or t_juged < t_pre else True
        return result

    def RecsDetection(self, save_loc, ftp):
        self.ins_list = []
        rot, rid, ext = self.__src.rot, self.__src.rid, self.__src.binfo.ex_t

        raw_lines = self.__ZifRead(rot, ftp)
        val_lines = [x for x in raw_lines if re.search(self.__patt, x)]
        raw_binfo = [list(re.findall(self.__patt, x)[0]) for x in val_lines]
        val_binfo = [x for x in raw_binfo if self.__TimeCheck(ext, x[1])]

        for binfo in val_binfo:
            rec_n, rec_t = binfo[0], self.TFtrans(binfo[1])
            rec_n_s = [rec_n + '.zdt', rec_n + '.zpx', rid + '.zif']

            zdt_check = not self._Basic__FileCheck(rot / rec_n_s[0], ftp)
            zpx_check = not self._Basic__FileCheck(rot / rec_n_s[1], ftp)

            if zdt_check or zpx_check:
                continue
            else:
                # zdt zpx file download
                dst_loc = save_loc / self.TPtrans(rec_t) / rid
                dst_loc.mkdir(parents=True, exist_ok=True)
                self._Basic__FileDownload(rot / rec_n_s[0], dst_loc / rec_n_s[0], ftp)
                self._Basic__FileDownload(rot / rec_n_s[1], dst_loc / rec_n_s[1], ftp)
                copy(self.__tmp, str(dst_loc / rec_n_s[2]))

                # result dict generate
                dict_ = self.__dst.copy()
                dict_['rid'], dict_['rec_t'] = rid, rec_t
                dict_['zdt'], dict_['zpx'] = rec_n, rec_n
                self.ins_list.append(dict_)

        remove(self.__tmp)

        if len(self.ins_list) < 1:
            self.__dst['rid'] = rid
            self.ins_list.append(self.__dst)