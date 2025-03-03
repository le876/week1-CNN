@echo off
echo 正在清理项目根目录...

REM 切换到根目录
cd ..

REM 删除已经复制到models目录的文件
if exist advanced_cnn_model.py del advanced_cnn_model.py
if exist compare_models.py del compare_models.py
if exist cnn_model.py del cnn_model.py

REM 删除已经复制到utils目录的文件
if exist data_processing.py del data_processing.py
if exist data_exploration.py del data_exploration.py
if exist check_file.py del check_file.py

REM 删除已经复制到analysis目录的文件
if exist model_visualization.py del model_visualization.py
if exist model_explanation.py del model_explanation.py
if exist hyperparameter_tuning.py del hyperparameter_tuning.py

REM 删除已经复制到scripts目录的文件
if exist run_pipeline.py del run_pipeline.py

REM 移动批处理文件到scripts目录
if exist setup_env.bat move setup_env.bat scripts\
if exist setup_anaconda_env.bat move setup_anaconda_env.bat scripts\
if exist activate_env.bat move activate_env.bat scripts\
if exist activate_powershell.bat move activate_powershell.bat scripts\
if exist run_without_activate.bat move run_without_activate.bat scripts\
if exist change_policy.bat move change_policy.bat scripts\

REM 移动数据文件到data目录
if exist *.npy move *.npy data\

REM 移动markdown文档到docs目录，保留README.md在根目录
if exist anaconda_guide.md move anaconda_guide.md docs\
if exist anaconda_navigator_guide.md move anaconda_navigator_guide.md docs\
if exist python_version_guide.md move python_version_guide.md docs\
if exist README_ANACONDA.md move README_ANACONDA.md docs\
if exist windows_env_guide.md move windows_env_guide.md docs\
if exist 解决PowerShell执行策略问题.md move 解决PowerShell执行策略问题.md docs\

echo 清理完成！根目录现在应该更加整洁。 