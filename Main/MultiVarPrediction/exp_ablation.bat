@ECHO OFF
setlocal enabledelayedexpansion
ECHO Ablation of foward search
PAUSE

set g_st_key[0]=basic_Nvar
set g_suffix[0]=gp_0_basic_Nvar

set g_st_key[1]=basic_var
set g_suffix[1]=gp_1_basic_var

set g_st_key[2]=basic_dis
set g_suffix[2]=gp_2_basic_dis

set g_st_key[3]=inds_Nvar
set g_suffix[3]=gp_3_inds_Nvar

set g_st_key[4]=inds_var
set g_suffix[4]=gp_4_inds_var

set g_st_key[5]=inds_dis
set g_suffix[5]=gp_5_inds_dis

set g_st_key[6]=mets_Nvar
set g_suffix[6]=gp_6_mets_Nvar

set g_st_key[7]=mets_var
set g_suffix[7]=gp_7_mets_var

set g_st_key[8]=all_Nvar
set g_suffix[8]=gp_8_all_Nvar

set g_st_key[9]=all_var
set g_suffix[9]=gp_9_all_var

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

set py_file=Main\MultiVarPrediction\main_forward_search.py

FOR /L %%i IN (9,1,9) DO (
    ECHO Start of Group %%i
    python %py_file% !g_suffix[%%i]! !g_st_key[%%i]!
    ECHO End of Processing !
    ping 127.0.0.1 -n 30 > nul
)

ECHO End of all exps!

PAUSE
EXIT