# 扩展技术文档

本文档为CNN函数逼近项目提供全面的技术细节。如需快速了解，请参见主[README.md](../README.md)。

## 项目概述

本项目展示了如何应用Convolutional Neural Networks来逼近复杂的数学函数（Ackley函数和Rosenbrock函数），构建了具备研究和生产应用高级特性的完整机器学习pipeline。

## 项目结构

- `data_exploration.py` - 数据集探索与分析
- `data_processing.py` - 数据预处理与增强
- `cnn_model.py` - 基础CNN模型定义和训练代码
- `advanced_cnn_model.py` - 高级CNN模型实现
- `compare_models.py` - 模型性能对比脚本
- `model_visualization.py` - 模型结构和特征可视化
- `model_explanation.py` - 模型解释工具（基于Captum）
- `hyperparameter_tuning.py` - 超参数调优（网格搜索和贝叶斯优化）
- `run_pipeline.py` - 完整训练流水线脚本
- `requirements.txt` - 项目依赖列表
- `setup_env.bat` - Windows系统下自动配置环境的批处理脚本

## 新增功能

与原始项目相比，本版本新增了以下功能：

1. **数据处理与增强**：
   - 数据标准化处理
   - 多种数据增强方法（噪声、抖动、缩放等）
   - 交叉验证数据集划分

2. **高级CNN模型**：
   - 更深层的网络结构
   - 批归一化层
   - 跳跃连接
   - 提前停止训练
   - 学习率调度

3. **模型可视化**：
   - 模型架构可视化
   - 卷积滤波器可视化
   - 特征图可视化
   - 二维预测空间可视化

4. **模型解释**：
   - 整合梯度解释算法
   - 遮挡解释算法
   - Layer GradCAM解释
   - 特征归因可视化

5. **超参数调优**：
   - 网格搜索优化
   - 贝叶斯优化
   - 超参数评估
   - 模型结果比较

6. **完整训练流水线**：
   - 命令行参数控制
   - 可定制的执行流程
   - 实验报告生成
   - 性能监控

## 快速开始

### 步骤1：配置环境

使用conda创建并激活一个新的环境：

```
conda create -n cnn_conda python=3.9
conda activate cnn_conda
```

安装依赖：

```
pip install -r requirements.txt
```

### 步骤2：运行数据探索

```
python data_exploration.py
```

### 步骤3：运行完整流水线

运行全部功能：

```
python run_pipeline.py
```

运行部分功能（例如跳过耗时的超参数优化）：

```
python run_pipeline.py --skip-hyperopt
```

使用快速模式：

```
python run_pipeline.py --quick-mode
```

启用数据增强：

```
python run_pipeline.py --augment --augmentation-factor 2
```

### 步骤4：查看结果

训练结果将保存在以下目录：

- `models/` - 保存训练好的模型
- `visualizations/` - 保存模型可视化结果
- `explanations/` - 保存模型解释结果
- `hyperopt/` - 保存超参数优化结果
- `processed_data/` - 保存处理后的数据
- `results/` - 保存实验报告

## 命令行参数

`run_pipeline.py`脚本支持以下命令行参数：

- `--skip-exploration` - 跳过数据探索步骤
- `--skip-processing` - 跳过数据处理步骤
- `--skip-basic-cnn` - 跳过基础CNN训练
- `--skip-advanced-cnn` - 跳过高级CNN训练
- `--skip-comparison` - 跳过模型比较
- `--skip-hyperopt` - 跳过超参数调优
- `--skip-visualization` - 跳过模型可视化
- `--skip-explanation` - 跳过模型解释
- `--quick-mode` - 使用快速模式（缩短训练和调优时间）
- `--augment` - 启用数据增强
- `--augmentation-factor N` - 设置数据增强因子（默认为1）

## 依赖项

项目的主要依赖包括：

- numpy
- matplotlib
- scipy
- scikit-learn
- torch
- torchvision
- bayesian-optimization
- captum
- torchviz

详细依赖请参见`requirements.txt`文件。

## 性能评估

模型性能使用以下指标评估：

1. 均方误差(MSE)
2. 皮尔逊相关系数
3. R²评分

## 进一步优化建议

如果需要进一步提高模型性能，可以尝试：

1. 增加更多数据增强方法
2. 尝试不同的网络架构（如Transformer或ResNet）
3. 使用集成学习方法
4. 添加更多正则化技术
5. 使用更高级的学习率调度策略 