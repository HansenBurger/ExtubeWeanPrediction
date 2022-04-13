from peewee import *
from playhouse.sqlite_ext import JSONField
from sqlalchemy import column

db = SqliteDatabase(r'C:\Main\Data\_\Database\sqlite\RespdataWean_2203.db')


class ExtubeBI(Model):

    pid = IntegerField(primary_key=True)
    icu = TextField()
    ex_t = DateTimeField(column_name='ExtubeTime')
    ex_s = TextField(column_name='ExtubeStatus')
    we_t = DateTimeField(column_name='WeanTime')
    we_s = TextField(column_name='WeanRemark')

    class Meta:
        table_name = 'ExtubeBasicInfo'
        database = db


class ZresParam(Model):

    index = AutoField()
    pid = IntegerField(column_name='patient_id')
    rid = TextField(column_name='record_id')
    rec_t = DateTimeField(column_name='record_time')

    class Meta:
        table_name = 'zres_param'
        database = db


class ZresDecode(Model):

    index = AutoField()
    pid = IntegerField(column_name='patient_id')
    rid = TextField(column_name='record_id')
    rec_t = DateTimeField(column_name='record_time')
    info = JSONField(column_name='resp_info')

    class Meta:
        table_name = 'zres_decode'
        database = db


class RidMatch(Model):

    pid = IntegerField(column_name='PID')
    rid = TextField(column_name='RID', primary_key=True)
    end_t = DateTimeField(column_name='END_t', null=True)
    rot = TextField(column_name='route', null=True)

    class Meta:
        table_name = 'RecordMatch_Id'
        database = db

    def InsertBlank(self):
        return {'pid': None, 'rid': None, 'end_t': None, 'rot': None}

    def __repr__(self) -> str:
        repr_ = 'pid: {0}, rid: {1}, rot: {2}'.format(self.pid, self.rid,
                                                      self.rot)
        return repr_


class RecMatch(Model):

    index = AutoField()
    rid = TextField(column_name='RID')
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='zdt_1', null=True)
    zpx = TextField(column_name='zpx_1', null=True)

    class Meta:
        table_name = 'RecordMatch_N'
        database = db

    def InsertBlank(self):
        return {'rid': None, 'rec_t': None, 'zdt': None, 'zpx': None}

    def __repr__(self) -> str:
        repr_ = 'rid: {0}, rec_t: {1}, zdt: {2}, zpx: {3}'.format(
            self.rid, self.rec_t, self.zdt, self.zpx)
        return repr_


class RecPreprocess(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    ex_t = DateTimeField(column_name='Extube_t')
    ex_s = TextField(column_name='Extube_end')
    rid = TextField(column_name='RID')
    end_t = DateTimeField(column_name='END_t', null=True)
    rec_t = DateTimeField(column_name='REC_t')
    zdt = TextField(column_name='zdt')
    zpx = TextField(column_name='zpx')

    class Meta:
        table_name = 'RecordPreprocess'
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
            'zpx': self.zpx
        }
        return dict_

    def __repr__(self) -> str:
        repr_ = 'pid:{0}, exTube:{1}, zdt:{2}, rec_t:{2}'.format(
            self.pid, self.ex_s, self.zdt, self.rec_t)
        return repr_