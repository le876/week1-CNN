# CNN函数逼近项目

一个综合性的深度学习项目，使用Convolutional Neural Networks (CNN)来逼近复杂的数学函数（Ackley函数和Rosenbrock函数），成功实现Pearson相关系数超过0.85的目标。

## 🎯 项目概述

本项目展示了如何应用CNN架构来学习和逼近高维数学函数。项目实现了完整的机器学习pipeline，包括数据处理、模型训练、超参数优化和综合评估体系。

### 核心成果
- **Rosenbrock函数**: Pearson相关系数 = 0.925, R² = 0.853
- **Ackley函数**: 使用多种CNN架构进行全面评估
- **健壮Pipeline**: 端到端自动化训练与评估系统

## 🏗️ 项目架构

```
├── data/                    # 训练和测试数据集
├── models/                  # CNN模型实现
│   ├── cnn_model.py        # 基础和高级CNN模型
│   ├── advanced_cnn_model.py # 带残差连接的增强CNN
│   └── compare_models.py   # 模型对比工具
├── utils/                   # 核心工具函数
│   ├── data_processing.py  # 数据预处理和增强
│   └── training.py         # 训练循环和评估
├── analysis/               # 分析和可视化工具
│   ├── model_visualization.py
│   ├── model_explanation.py
│   └── hyperparameter_tuning.py
├── scripts/                # 执行脚本
│   └── run_pipeline.py     # 主训练pipeline
└── results/                # 训练输出和可视化结果
```

## ✨ 核心特性

### 🔧 高级数据处理
- **智能预处理**: 特征标准化和2D重塑，适配CNN输入要求
- **数据增强**: 高斯噪声、随机缩放和合成样本生成，提升模型泛化能力
- **健壮验证**: 训练/验证/测试划分，支持交叉验证策略

### 🧠 多种CNN架构
- **基础CNN**: 高效的2层架构，集成batch normalization
- **高级CNN**: 深度网络，包含残差连接和注意力机制
- **增强CNN**: 多尺度特征提取，配备高级正则化技术

### 📊 全面评估体系
- **多指标评估**: Pearson相关系数、MSE、RMSE、MAE、R²
- **归一化指标**: 尺度不变的评估方法，确保公平比较
- **可视化分析**: 训练曲线、预测图表、误差分布

### 🔍 模型可解释性
- **特征可视化**: 卷积滤波器和激活图展示
- **归因分析**: Integrated Gradients、Occlusion、GradCAM
- **架构洞察**: 逐层特征提取分析

## 🚀 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd week1_CNN
```

2. **创建并激活虚拟环境**
```bash
conda create -n cnn_function_approx python=3.9
conda activate cnn_function_approx
```

3. **安装依赖包**
```bash
pip install -r requirements.txt
```

### 基本使用

1. **运行完整pipeline**
```bash
python scripts/run_pipeline.py
```

2. **快速训练模式（减少训练轮数）**
```bash
python scripts/run_pipeline.py --quick-mode
```

3. **启用数据增强训练**
```bash
python scripts/run_pipeline.py --augment --augmentation-factor 2
```

4. **跳过特定组件**
```bash
python scripts/run_pipeline.py --skip-hyperopt --skip-visualization
```

### 输出结构
```
results/
├── {function}_{model}_model.pth      # 训练好的模型
├── {function}_{model}_results.txt    # 性能指标
├── {function}_{model}_history.png    # 训练曲线
├── {function}_{model}_predictions.png # 预测结果图
└── visualization/                    # 模型分析
```

## 📈 性能结果

### Rosenbrock函数逼近
- **Pearson相关系数**: 0.925
- **R²分数**: 0.853
- **MSE**: 915,819,392 (归一化: 0.006)
- **架构**: 基础CNN配合batch normalization

### Ackley函数逼近
- **多架构测试**: 基础、高级、增强型CNN
- **全面评估**: 多指标评估和可视化分析
- **稳定性能**: 在不同配置下保持一致的结果

## 🛠️ 技术细节

### CNN架构设计
```python
BasicCNN(
  (conv1): Conv2d(1, 16, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(16)
  (pool1): MaxPool2d(kernel_size=2, stride=1)
  (conv2): Conv2d(16, 32, kernel_size=2, padding=1)
  (bn2): BatchNorm2d(32)
  (fc1): Linear(640, 64)
  (dropout): Dropout(0.3)
  (fc2): Linear(64, 1)
)
```

### 数据处理Pipeline
1. **输入变换**: 20维特征向量 → 4×5 2D矩阵
2. **标准化**: 零均值单位方差归一化
3. **数据增强**: 高斯噪声 + 随机缩放
4. **验证划分**: 80/20训练验证比例

### 评估指标体系
- **Pearson相关系数**: 主要性能指标
- **均方误差(MSE)**: 原始值和归一化版本
- **R²决定系数**: 模型解释能力评估
- **RMSE & MAE**: 额外的回归评估指标

## 📚 文档说明

- [`CNN_架构.md`](CNN_架构.md) - 详细的CNN架构技术说明
- [`docs/README.md`](docs/README.md) - 扩展技术文档
- [`analysis/`](analysis/) - 模型分析和可视化工具

## 🔧 命令行选项

| 选项 | 说明 |
|------|------|
| `--quick-mode` | 快速模式，减少训练轮数用于快速测试 |
| `--augment` | 启用数据增强功能 |
| `--augmentation-factor N` | 设置数据增强倍数 |
| `--skip-hyperopt` | 跳过超参数优化步骤 |
| `--skip-visualization` | 跳过模型可视化步骤 |
| `--skip-explanation` | 跳过模型解释分析步骤 |

## 📦 依赖环境

核心依赖包:
- `torch>=2.0.1` - 深度学习框架
- `numpy>=1.24.4` - 数值计算库
- `scikit-learn>=1.3.0` - 机器学习工具包
- `matplotlib>=3.7.2` - 数据可视化库
- `captum>=0.6.0` - 模型可解释性工具

完整依赖列表请参见 [`requirements.txt`](requirements.txt)。

## 🤝 贡献指南

本项目展示了CNN在数学函数逼近中的应用。欢迎进行以下改进:
- 尝试不同的网络架构设计
- 测试更多数学函数类型
- 改进可视化和分析工具
- 优化超参数配置

## 📄 许可证

本项目仅用于教育和研究目的。