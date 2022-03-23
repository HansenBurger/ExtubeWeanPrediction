import pandas as pd
from pathlib import Path
from Classes.Func import kit
from Classes.Domain import layer_0, layer_1, layer_2
from WaveDataProcess import BinImport

df = pd.read_csv('1.csv')
kit.TimeShift(df, ['endo_t', 'END_t', 'Resp_t'])
gp = df.groupby('PID')


def GetGp(pid):
    df = gp.get_group(pid)
    return df


df_ = GetGp(2317017)
df_ = df_.reset_index(drop=True)
df_ = df_.loc[::-1].reset_index(drop=True)
pid = df_.PID.unique().item()
icu = df_.ICU.unique().item()
end_t = df_.endo_t[0]  # get by row not unique
end_i = df_.endo_end.unique().item()
rid_s = df_.Record_id.unique()[0]
result = None

file_loc = Path(kit.ConfigRead('WaveData', 'Extube'))

recid = layer_2.RidRec()
recid.zif = file_loc / (str(end_t.year) + str(end_t.month).rjust(
    2, '0')) / rid_s / (rid_s + '.zif')
recid.cls_t = pd.DataFrame(BinImport.RidData(
    recid.zif).RecordListGet()).s_t.max()
recid.vm_n = BinImport.RidData(recid.zif).RecordInfoGet()['m_n']

import instantiation as instan

wave_p = instan.RecordResp(recid.zif.parent, df_.zdt_1[0])
wave_p.WaveformInit()
wave_p.IndicatorCalculate()
rec_wave = wave_p.rec
para_p = instan.RecordPara(recid.zif.parent, df_.zpx_1[0])
para_p.ParametersInit(recid.vm_n)
para_p.ParaSelectBT(rec_wave.sr, [0, 600, 1800], para_p.rec.st_mode)
