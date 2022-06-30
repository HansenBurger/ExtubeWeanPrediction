@ECHO OFF
ECHO Download, Verify, Filt the RespDatas in the OutCome of extube and wean table
PAUSE

set t_st_0=1800
set n_st_0=VarAnalysis_30min
set t_st_1=3600
set n_st_1=VarAnalysis_60min

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\ExWeanPredict
cd C:\Main\Project\ExtubeWeanPrediction

ECHO Start of variability analysis 30min before extubation!
python Main\RespDataPreprocess\0_download.py %t_st_0% %n_st_0%
ECHO End of Processing !

:: sleep 30 s for wean process
ping 127.0.0.1 -n 30 > nul

ECHO Wean data process start!
ECHO Start of variability analysis 60min before extubation!
python Main\RespDataPreprocess\0_download.py %t_st_1% %n_st_1%
ECHO End of Process
ECHO End of Processing !

PAUSE
EXIT