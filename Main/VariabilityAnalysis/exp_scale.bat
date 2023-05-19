@ECHO OFF
setlocal enabledelayedexpansion
ECHO Calculation of respiratory variability before operation

set t_st=3600

@REM set s_st_s[0]=60
@REM set n_st_s[0]=VarAnalysis_60m_1m

@REM set s_st_s[1]=180
@REM set n_st_s[1]=VarAnalysis_60m_3m

set s_st_s[0]=4
set n_st_s[0]=VarAnalysis_60m_4s

set s_st_s[1]=8
set n_st_s[1]=VarAnalysis_60m_8s

set s_st_s[2]=16
set n_st_s[2]=VarAnalysis_60m_16s

set s_st_s[3]=32
set n_st_s[3]=VarAnalysis_60m_32s

set s_st_s[4]=64
set n_st_s[4]=VarAnalysis_60m_64s

set s_st_s[5]=128
set n_st_s[5]=VarAnalysis_60m_128s

set s_st_s[6]=2700
set n_st_s[6]=VarAnalysis_60m_45m

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

if %conda_env%==q EXIT

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

set py_file=Main\VariabilityAnalysis\main.py

FOR /L %%i IN (5,1,5) DO (
    CALL ECHO Start of %%n_st_s[%%i]%%
    :: main program running
    python %py_file% !t_st! !n_st_s[%%i]! !s_st_s[%%i]!
    ECHO End of Processing !
    ping 127.0.0.1 -n 30 > nul
)

ECHO End of all !

PAUSE
EXIT