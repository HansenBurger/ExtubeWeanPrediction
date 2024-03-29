@ECHO OFF
ECHO Download, Verify, Filt the RespDatas in the OutCome of extube and wean table
PAUSE

set mode_1=Extube
set mode_2=Wean

ECHO.

set /p conda_env=Enter the virtual environment name[q:quit] :

:: activate the process env
call C:\Main\Soft\Conda\Scripts\activate.bat C:\Main\Soft\Conda\envs\%conda_env%
cd C:\Main\Project\ExtubeWeanPrediction

ECHO Extube data process start!
ECHO Step1: Data download
python Main\RespDataPreprocess\0_download.py %mode_1%
ECHO Step2: Records Verify
python Main\RespDataPreprocess\1_verify.py %mode_1%
ECHO Step3: Filt by conditions
python Main\RespDataPreprocess\2_condfilt.py %mode_1%
ECHO Extube data process end!

:: sleep 30 s for wean process
@REM ping 127.0.0.1 -n 30 > nul

@REM ECHO Wean data process start!
@REM ECHO Step1: Data download
@REM python Main\RespDataPreprocess\0_download.py %mode_2%
@REM ECHO Step2: Records Verify
@REM python Main\RespDataPreprocess\1_verify.py %mode_2%
@REM ECHO Step3: Filt by conditions
@REM python Main\RespDataPreprocess\2_condfilt.py %mode_2%
@REM ECHO Wean data process end!

PAUSE
EXIT