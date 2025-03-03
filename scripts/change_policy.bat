@echo off
echo ===================================================
echo    永久更改PowerShell执行策略（需要管理员权限）
echo ===================================================
echo.

echo 此脚本将永久更改PowerShell执行策略，允许运行本地脚本
echo 需要管理员权限才能执行此操作
echo.
echo 按任意键继续...
pause > nul

REM 以管理员身份运行PowerShell并永久更改执行策略
powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser; Write-Host \"执行策略已更改为RemoteSigned\"; cd \"%~dp0\"; .\\cnn_env\\Scripts\\Activate.ps1'" -Verb RunAs 