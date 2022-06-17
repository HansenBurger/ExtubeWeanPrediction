import sys
from peewee import *
from pathlib import Path

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import ConfigRead

db = SqliteDatabase(ConfigRead('RespData', 'Main'))


class PatientInfo(Model):

    pid = IntegerField(column_name='PID', primary_key=True)
    age = FloatField(column_name='age', null=True)
    sex = IntegerField(column_name='sex', null=True)
    bmi = FloatField(column_name='BMI', null=True)
    rmk = TextField(column_name='remark', null=True)
    rmk_i = TextField(column_name='RemarkIn', null=True)
    rmk_o = TextField(column_name='RemarkOut', null=True)
    rmk_i_ = TextField(column_name='RemarkInICU', null=True)
    rmk_o_ = TextField(column_name='RemarkOutICU', null=True)
    rmk_t = TextField(column_name='RemarkType', null=True)

    class Meta:
        table_name = 'PatientInfo_Total'
        database = db


class LabExtube(Model):

    pid = IntegerField(column_name='PID', primary_key=True)
    # b_hr = FloatField(column_name='bed_HR', null=True)
    # b_sbp = FloatField(column_name='bed_SBP', null=True)
    # b_dbp = FloatField(column_name='bed_DBP', null=True)
    # b_mbp = FloatField(column_name='bed_MBP', null=True)
    # b_spo = FloatField(column_name='bed_SpO2', null=True)
    be = FloatField(column_name='BE', null=True)
    co2 = FloatField(column_name='CO2', null=True)
    ct = FloatField(column_name='CT', null=True)
    cl = FloatField(column_name='Cl', null=True)
    fio2 = FloatField(column_name='FiO2', null=True)
    ga = FloatField(column_name='Ga', null=True)
    hco3 = FloatField(column_name='HCO3', null=True)
    hemo = FloatField(column_name='Hemo', null=True)
    K = FloatField(column_name='K', null=True)
    lac = FloatField(column_name='Lac', null=True)
    mhb = FloatField(column_name='MHb', null=True)
    na = FloatField(column_name='Na', null=True)
    nph = FloatField(column_name='NpH', null=True)
    npco2 = FloatField(column_name='NpaCO2', null=True)
    npo2 = FloatField(column_name='NpaO2', null=True)
    o2 = FloatField(column_name='O2', null=True)
    osp = FloatField(column_name='OsP', null=True)
    pcv = FloatField(column_name='PCV', null=True)
    pf = FloatField(column_name='PF', null=True)
    po2 = FloatField(column_name='PaO2', null=True)
    rhs = FloatField(column_name='RHS', null=True)
    tbil = FloatField(column_name='TBIL', null=True)
    temp = FloatField(column_name='Temp', null=True)
    wbg = FloatField(column_name='WBG', null=True)
    cr = FloatField(column_name='肌酐', null=True)
    crp = FloatField(column_name='超敏C反应蛋白', null=True)
    sofa = FloatField(column_name='sofa', null=True)
    apache = FloatField(column_name='apache', null=True)

    class Meta:
        table_name = 'Laboratory_Extube'
        database = db


class LabWean(Model):

    pid = IntegerField(column_name='PID', primary_key=True)
    # b_hr = FloatField(column_name='bed_HR', null=True)
    # b_sbp = FloatField(column_name='bed_SBP', null=True)
    # b_dbp = FloatField(column_name='bed_DBP', null=True)
    # b_mbp = FloatField(column_name='bed_MBP', null=True)
    # b_spo = FloatField(column_name='bed_SpO2', null=True)
    be = FloatField(column_name='BE', null=True)
    co2 = FloatField(column_name='CO2', null=True)
    ct = FloatField(column_name='CT', null=True)
    cl = FloatField(column_name='Cl', null=True)
    fio2 = FloatField(column_name='FiO2', null=True)
    ga = FloatField(column_name='Ga', null=True)
    hco3 = FloatField(column_name='HCO3', null=True)
    hemo = FloatField(column_name='Hemo', null=True)
    K = FloatField(column_name='K', null=True)
    lac = FloatField(column_name='Lac', null=True)
    mhb = FloatField(column_name='MHb', null=True)
    na = FloatField(column_name='Na', null=True)
    nph = FloatField(column_name='NpH', null=True)
    npco2 = FloatField(column_name='NpaCO2', null=True)
    npo2 = FloatField(column_name='NpaO2', null=True)
    o2 = FloatField(column_name='O2', null=True)
    osp = FloatField(column_name='OsP', null=True)
    pcv = FloatField(column_name='PCV', null=True)
    pf = FloatField(column_name='PF', null=True)
    po2 = FloatField(column_name='PaO2', null=True)
    rhs = FloatField(column_name='RHS', null=True)
    tbil = FloatField(column_name='TBIL', null=True)
    temp = FloatField(column_name='Temp', null=True)
    wbg = FloatField(column_name='WBG', null=True)
    cr = FloatField(column_name='肌酐', null=True)
    crp = FloatField(column_name='超敏C反应蛋白', null=True)
    sofa = FloatField(column_name='sofa', null=True)
    apache = FloatField(column_name='apache', null=True)

    class Meta:
        table_name = 'Laboratory_Wean'
        database = db
