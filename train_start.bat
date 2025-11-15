@echo off
REM train_start.bat — Start training and save logs. Resumes if checkpoint exists.
cd /d %~dp0

echo Working directory: %cd%

REM Activate virtualenv if it exists
if exist "venv\Scripts\activate.bat" (
  call "venv\Scripts\activate.bat"
  echo Activated virtualenv: venv
) else (
  echo No virtualenv found. Using system Python.
)

echo Starting training at %date% %time%
REM Run training, unbuffered output, redirect stdout+stderr to train.log
python -u train.py > train.log 2>&1
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
  echo Training exited with error %EXITCODE%. See train.log for details.
) else (
  echo Training finished successfully.
)

echo Checkpoint: model\checkpoint.pth
echo Logs: train.log
pause
