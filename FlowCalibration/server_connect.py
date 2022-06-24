import sys, pathlib
from ftplib import FTP

sys.path.append(pathlib.Path.cwd())

from Classes.Func.KitTools import ConfigRead


class MYFTP():
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


def main():
    serv_i = ConfigRead('Server')
    ftp_ = MYFTP(**serv_i)
    ftp_.FtpLogin()
    ftp_.FtpLogout()


if __name__ == '__main__':
    main()