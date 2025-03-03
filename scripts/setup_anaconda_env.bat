@echo off
echo ===================================================
echo    Anaconda CNN项目环境配置工具
echo ===================================================
echo.

REM 检查Anaconda是否安装
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Anaconda。请确保已安装Anaconda并添加到PATH中。
    echo 您可以从开始菜单打开"Anaconda Prompt"，然后手动运行命令。
    pause
    exit /b
)

echo [信息] 检测到Anaconda安装
echo.

REM 创建环境
echo [信息] 创建名为cnn_env的Python 3.9环境...
call conda create -n cnn_env python=3.9 -y
if %errorlevel% neq 0 (
    echo [错误] 创建环境失败
    pause
    exit /b
)
echo.

REM 激活环境
echo [信息] 激活环境...
call conda activate cnn_env
if %errorlevel% neq 0 (
    echo [错误] 激活环境失败
    pause
    exit /b
)
echo.

REM 安装基本科学计算包
echo [信息] 安装基本科学计算包...
call conda install numpy matplotlib scipy scikit-learn -y
if %errorlevel% neq 0 (
    echo [警告] 安装基本包时出现错误
)
echo.

REM 安装PyTorch
echo [信息] 安装PyTorch (CPU版本)...
call conda install pytorch torchvision cpuonly -c pytorch -y
if %errorlevel% neq 0 (
    echo [警告] 安装PyTorch时出现错误
)
echo.

REM 验证安装
echo [信息] 验证安装...
python -c "import numpy, matplotlib, scipy, sklearn, torch; print('所有包导入成功!'); print('PyTorch版本:', torch.__version__)"
if %errorlevel% neq 0 (
    echo [警告] 验证失败，某些包可能未正确安装
) else (
    echo [成功] 所有包验证通过
)
echo.

REM 导出环境配置
echo [信息] 导出环境配置...
call conda env export > cnn_environment.yml
echo [成功] 环境配置已保存到 cnn_environment.yml
echo.

echo ===================================================
echo    环境设置完成!
echo ===================================================
echo.
echo 使用说明:
echo 1. 每次使用前，请先激活环境:
echo    conda activate cnn_env
echo.
echo 2. 运行CNN模型:
echo    python cnn_model.py
echo.
echo 3. 完成后，可以退出环境:
echo    conda deactivate
echo.
echo 4. 如需删除环境（释放空间）:
echo    conda env remove -n cnn_env
echo ===================================================

REM 保持环境激活状态
cmd /k "conda activate cnn_env && cd /d %~dp0" 