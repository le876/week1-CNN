@echo off
echo ===================================================
echo    不激活环境直接运行Python（使用虚拟环境中的Python）
echo ===================================================
echo.

REM 直接使用虚拟环境中的Python解释器
set PYTHON_PATH=%~dp0cnn_env\Scripts\python.exe

echo 使用Python: %PYTHON_PATH%
echo.

REM 运行CNN模型
echo 运行CNN模型...
"%PYTHON_PATH%" "%~dp0cnn_model.py"

echo.
echo 按任意键退出...
pause > nul 