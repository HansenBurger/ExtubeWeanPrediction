import sys
from peewee import *
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead

db = SqliteDatabase(ConfigRead('RespData', 'Main'))


class ExtubePSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Extube_PSV'
        database = db


class ExtubeNotPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Extube_NotPSV'
        database = db


class ExtubeSumP12(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Extube_SumP12'
        database = db


class WeanPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Wean_PSV'
        database = db


class WeanNotPSV(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Wean_NotPSV'
        database = db


class WeanSumP12(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)

    class Meta:
        table_name = 'Wean_SumP12'
        database = db