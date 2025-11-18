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
REM Run training and both display stdout and save to train.log using PowerShell Tee-Object.
REM Use the venv python executable directly so the environment is consistent.
REM Detect a sample image in .\test_image and pass it to train.py via --sample_image
set "SAMPLE_IMAGE="
for %%p in (test_image\*.png test_image\*.jpg test_image\*.jpeg test_image\*.webp) do (
	set "SAMPLE_IMAGE=%%~dpnxp"
	goto :found_sample
)
:found_sample
if defined SAMPLE_IMAGE (
	echo Found sample image: %SAMPLE_IMAGE%
	powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '.\\.venv\\Scripts\\python.exe' -u 'train.py' --sample_image '%SAMPLE_IMAGE%' 2>&1 | Tee-Object -FilePath 'train.log'; exit $LASTEXITCODE }"
) else (
	echo No sample image found in .\test_image; running without sample_image
	powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '.\\.venv\\Scripts\\python.exe' -u 'train.py' 2>&1 | Tee-Object -FilePath 'train.log'; exit $LASTEXITCODE }"
)
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
	echo Training exited with error %EXITCODE%. See train.log for details.
) else (
	echo Training finished successfully.
)

echo Checkpoint: model\checkpoint.pth
echo Logs: train.log
pause
