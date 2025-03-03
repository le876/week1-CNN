@echo off
echo ===================================================
echo    使用CMD激活虚拟环境
echo ===================================================
echo.

REM 打开CMD并激活环境
cmd /k "cd /d %~dp0 && cnn_env\Scripts\activate.bat" 