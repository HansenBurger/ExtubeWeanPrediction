import sys
from peewee import *
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead

db = SqliteDatabase(ConfigRead('RespData', 'Main'))


class ExtubePSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    def ObjToDict(self):
        dict_ = {
            'pid': self.pid,
            'icu': self.icu,
            'e_t': self.e_t,
            'e_s': self.e_s,
            'rid': self.rid,
            'rec_t': self.rec_t,
            'zdt': self.zdt,
            'zpx': self.zpx,
            'opt': self.opt,
        }
        return dict_

    class Meta:
        table_name = 'Extube_PSV'
        database = db


class ExtubeNotPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Extube_NotPSV'
        database = db


class ExtubeSumP10(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Extube_SumP10'
        database = db


class ExtubeSumP12(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Extube_SumP12'
        database = db


class WeanPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Wean_PSV'
        database = db


class WeanNotPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Wean_NotPSV'
        database = db


class WeanSumP10(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Wean_SumP10'
        database = db


class WeanSumP12(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    opt = BooleanField(column_name='op_tag')

    class Meta:
        table_name = 'Wean_SumP12'
        database = db