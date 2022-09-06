@ECHO OFF
setlocal enabledelayedexpansion
ECHO Calculation of respiratory variability before operation

set t_st_s[0]=1800
set n_st_s[0]=VarAnalysis_30_min

set t_st_s[1]=3600
set n_st_s[1]=VarAnalysis_60_min

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

if %conda_env%==q EXIT

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

set py_file=Main\VariabilityAnalysis\main.py

FOR /L %%i IN (1,1,1) DO (
    CALL ECHO Start of %%n_st_s[%%i]%%
    :: main program running
    python %py_file% !t_st_s[%%i]! !n_st_s[%%i]!
    ECHO End of Processing !
    ping 127.0.0.1 -n 30 > nul
)

PAUSE
EXIT