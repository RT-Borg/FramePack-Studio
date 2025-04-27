@echo off
echo Starting FramePack-Studio...

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Cannot run studio.py.
    goto end
)

REM Run the studio.py script using the python interpreter
python studio.py

REM Check the exit code of the python script
if %errorlevel% neq 0 (
    echo FramePack-Studio finished with an error.
) else (
    echo FramePack-Studio finished successfully.
)

:end
echo Exiting run script.
pause
