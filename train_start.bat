@echo off
REM train_start.bat — create venv, install dependencies, then run training
cd /d %~dp0

REM Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate" (
	python -m venv .venv
	echo Created virtual environment .venv
)

REM Activate virtual environment
call .\.venv\Scripts\activate

REM Upgrade pip and wheel
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch (CUDA 12.8 wheel). Adjust index-url if your CUDA/toolkit differs.
echo Installing PyTorch (cuda-enabled). If you need CPU-only, edit this file.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio --extra-index-url https://pypi.org/simple

REM Install other required Python packages
pip install -r requirements.txt

REM Ensure model directory exists
if not exist model mkdir model

echo Starting training at %date% %time%
REM Run training and save logs
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
