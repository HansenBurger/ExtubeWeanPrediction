@ECHO OFF
setlocal enabledelayedexpansion
ECHO Varehence of foward search
PAUSE

set g_feat_way[0]=var
set g_st_key[0]=exp_best
set g_suffix[0]=GP_NVar

set g_feat_way[1]=lab
set g_st_key[1]=exp_best
set g_suffix[1]=GP_Lab

set g_feat_way[2]=all
set g_st_key[2]=exp_best
set g_suffix[2]=GP_all

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

set py_file=Main\MultiVarPrediction\main_forward_search.py

FOR /L %%i IN (0,1,2) DO (
    ECHO Start of Group %%i
    python %py_file% !g_suffix[%%i]! !g_st_key[%%i]! !g_feat_way[%%i]!
    ECHO End of Processing !
    ping 127.0.0.1 -n 30 > nul
)

ECHO End of all !
PAUSE
EXIT