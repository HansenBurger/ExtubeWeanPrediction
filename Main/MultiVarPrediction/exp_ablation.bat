@ECHO OFF
setlocal enabledelayedexpansion
ECHO Ablation of foward search
PAUSE

set g_st[0].st_key = basic_Nvar
set g_st[0].suffix = gp_0_basic_Nvar

set g_st[1].st_key = basic_var
set g_st[1].suffix = gp_1_basic_var

set g_st[2].st_key = basic_dis
set g_st[2].suffix = gp_2_basic_dis

set g_st[3].st_key = inds_Nvar
set g_st[3].suffix = gp_3_inds_Nvar

set g_st[4].st_key = inds_var
set g_st[4].suffix = gp_4_inds_var

set g_st[5].st_key = inds_dis
set g_st[5].suffix = gp_5_inds_dis

set g_st[6].st_key = mets_Nvar
set g_st[6].suffix = gp_6_mets_Nvar

set g_st[7].st_key = mets_var
set g_st[7].suffix = gp_7_mets_var

set g_st[8].st_key = all_Nvar
set g_st[8].suffix = gp_8_all_Nvar

set g_st[9].st_key = all_var
set g_st[9].suffix = gp_9_all_var

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

set py_file=Main\MultiVarPrediction\main_forward_search.py

FOR /L %%i IN (0,1,9) DO (
    ECHO Start of Group %%i
    python %py_file% !g_st[%%i].suffix! !g_st[%%i].st_key!
    ECHO End of Processing !
    ping 127.0.0.1 -n 30 > nul
)

ECHO End of all exps!

PAUSE
EXIT