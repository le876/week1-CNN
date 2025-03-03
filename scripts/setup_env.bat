@echo off
echo ===================================================
echo    CNN项目环境配置工具 (Windows系统)
echo ===================================================
echo.

REM 检查Python版本
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python。请先安装Python 3.9
    echo 请访问: https://www.python.org/downloads/release/python-3913/
    echo 安装时请勾选"Add Python to PATH"选项
    pause
    exit /b
)

REM 显示Python版本
echo [信息] 检测到Python:
python --version
echo.

REM 创建项目文件夹（如果不存在）
if not exist "cnn_project" (
    echo [信息] 创建项目文件夹...
    mkdir cnn_project
)

REM 切换到项目文件夹
cd cnn_project
echo [信息] 当前工作目录: %cd%
echo.

REM 创建虚拟环境
echo [信息] 创建虚拟环境...
if exist "cnn_env" (
    echo [警告] 虚拟环境已存在，跳过创建步骤
) else (
    python -m venv cnn_env
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b
    )
    echo [成功] 虚拟环境创建完成
)
echo.

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call cnn_env\Scripts\activate
if %errorlevel% neq 0 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b
)
echo [成功] 虚拟环境已激活
echo.

REM 创建requirements.txt文件
echo [信息] 创建依赖文件...
echo numpy==1.22.4 > requirements.txt
echo matplotlib==3.5.2 >> requirements.txt
echo torch==2.0.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu >> requirements.txt
echo torchvision==0.15.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu >> requirements.txt
echo scikit-learn==1.1.2 >> requirements.txt
echo scipy==1.8.1 >> requirements.txt
echo [成功] 依赖文件已创建
echo.

REM 升级pip
echo [信息] 升级pip...
python -m pip install --upgrade pip
echo.

REM 安装依赖
echo [信息] 安装必要的包（这可能需要几分钟时间）...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [警告] 安装包时出现错误，请检查网络连接
    pause
) else (
    echo [成功] 所有包安装完成
)
echo.

REM 验证安装
echo [信息] 验证安装...
python -c "import numpy, matplotlib, torch, torchvision, sklearn, scipy; print('所有包导入成功!')" > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 验证失败，某些包可能未正确安装
) else (
    echo [成功] 所有包验证通过
)
echo.

REM 复制模型文件（如果存在）
if exist "..\cnn_model.py" (
    echo [信息] 复制CNN模型文件...
    copy ..\cnn_model.py .
    echo [成功] 模型文件已复制
)
echo.

echo ===================================================
echo    环境设置完成!
echo ===================================================
echo.
echo 使用说明:
echo 1. 每次使用前，请先激活环境:
echo    call cnn_env\Scripts\activate
echo.
echo 2. 运行CNN模型:
echo    python cnn_model.py
echo.
echo 3. 完成后，可以退出环境:
echo    deactivate
echo.
echo 4. 如需删除环境（释放空间）:
echo    rmdir /s /q cnn_env
echo ===================================================

pause 