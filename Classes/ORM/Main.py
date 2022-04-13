from peewee import *
from playhouse.sqlite_ext import JSONField
from sqlalchemy import null

db = SqliteDatabase(r'C:\Main\Data\_\Database\sqlite\RespdataWean_2203.db')


class ZresDecode(Model):

    index = AutoField()
    pid = IntegerField(column_name='patient_id')
    rid = TextField(column_name='record_id')
    rec_t = DateTimeField(column_name='record_time')
    info = JSONField(column_name='resp_info')

    class Meta:
        table_name = 'zres_decode'
        database = db


class ExtubePrep(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    end_t = DateTimeField(column_name='END_t')
    end_i = TextField(column_name='END_i')
    rid = TextField(column_name='RID')
    tail_t = DateTimeField(column_name='TAIL_t', null=True)
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    vme = TextField(column_name='vm_end', null=True)
    vmd = JSONField(column_name='vm_dict', null=True)
    spd = JSONField(column_name='sump', null=True)

    class Meta:
        table_name = 'ExtubePrep'
        database = db

    def ObjToDict(self):
        dict_ = {
            'pid': self.pid,
            'icu': self.icu,
            'ex_t': self.ex_t,
            'ex_s': self.ex_s,
            'rid': self.rid,
            'end_t': self.end_t,
            'rec_t': self.rec_t,
            'zdt': self.zdt,
            'zpx': self.zpx,
            'vme': self.vme,
            'vmd': self.vmd,
            'spd': self.spd
        }
        return dict_


class WeanPrep(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    end_t = DateTimeField(column_name='END_t')
    end_i = TextField(column_name='END_i')
    rid = TextField(column_name='RID')
    tail_t = DateTimeField(column_name='TAIL_t', null=True)
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='ZDT')
    zpx = TextField(column_name='ZPX')
    vme = TextField(column_name='vm_end', null=True)
    vmd = JSONField(column_name='vm_dict', null=True)
    spd = JSONField(column_name='sump', null=True)

    class Meta:
        table_name = 'WeanPrep'
        database = db

    def ObjToDict(self):
        dict_ = {
            'pid': self.pid,
            'icu': self.icu,
            'ex_t': self.ex_t,
            'ex_s': self.ex_s,
            'rid': self.rid,
            'end_t': self.end_t,
            'rec_t': self.rec_t,
            'zdt': self.zdt,
            'zpx': self.zpx,
            'vme': self.vme,
            'vmd': self.vmd,
            'spd': self.spd
        }
        return dict_