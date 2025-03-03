@echo off
echo ===================================================
echo    临时更改PowerShell执行策略并激活环境
echo ===================================================
echo.

REM 以管理员身份运行PowerShell并临时更改执行策略
powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', 'cd \"%~dp0\"; .\\cnn_env\\Scripts\\Activate.ps1'" -Verb RunAs 