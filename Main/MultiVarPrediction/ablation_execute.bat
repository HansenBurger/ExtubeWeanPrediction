@ECHO OFF
ECHO Ablation of foward search
PAUSE

:: Group 1 - Origin
set g_n_0=group_1
set n_s_0=gp_origin

:: Group 2 - Add MP, WOB
set g_n_1=group_2
set n_s_1=gp_indicators

:: Group 3 - Add other variability methods
set g_n_2=group_3
set n_s_2=gp_methods

:: Group 4 - Add All
set g_n_3=group_4
set n_s_3=gp_all

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\DF_process
cd C:\Main\Project\ExtubeWeanPrediction

ECHO Start of Group_1 foward test !
python Main\MultiVarPrediction\main_forward_search.py %g_n_0% %n_s_0%
ECHO End of Processing !

:: sleep 30 s for next group
ping 127.0.0.1 -n 30 > nul

ECHO Start of Group_2 foward test !
python Main\MultiVarPrediction\main_forward_search.py %g_n_1% %n_s_1%
ECHO End of Processing !

ping 127.0.0.1 -n 30 > nul

ECHO Start of Group_3 foward test !
python Main\MultiVarPrediction\main_forward_search.py %g_n_2% %n_s_2%
ECHO End of Processing !

ping 127.0.0.1 -n 30 > nul

ECHO Start of Group_4 foward test !
python Main\MultiVarPrediction\main_forward_search.py %g_n_3% %n_s_3%
ECHO End of Processing !

ECHO End of All !

PAUSE
EXIT