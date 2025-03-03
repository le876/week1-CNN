@echo off
echo 整理项目文件...

REM 先删除旧目录，然后创建新目录
rmdir /s /q models data utils scripts docs analysis 2>nul
mkdir models data utils scripts docs analysis

REM 移动模型文件到models目录
copy advanced_cnn_model.py models\ >nul
copy compare_models.py models\ >nul
REM 尝试复制可能不存在的文件
if exist cnn_model.py copy cnn_model.py models\ >nul

REM 移动数据处理文件到utils目录
copy data_processing.py utils\ >nul
copy data_exploration.py utils\ >nul
copy check_file.py utils\ >nul

REM 移动模型分析文件到analysis目录
copy model_visualization.py analysis\ >nul
copy model_explanation.py analysis\ >nul
copy hyperparameter_tuning.py analysis\ >nul

REM 移动主流程文件到scripts目录
copy run_pipeline.py scripts\ >nul
copy organize_files.bat scripts\ >nul
copy setup_env.bat scripts\ >nul
copy setup_anaconda_env.bat scripts\ >nul
copy activate_env.bat scripts\ >nul
copy activate_powershell.bat scripts\ >nul
copy run_without_activate.bat scripts\ >nul
copy change_policy.bat scripts\ >nul

REM 移动文档文件到docs目录
copy *.md docs\ >nul

REM 移动数据文件到data目录
copy *.npy data\ >nul

echo 文件整理完成！ 