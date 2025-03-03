import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
import os
import sys

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from models.cnn_model import CNN, load_data
from models.advanced_cnn_model import AdvancedCNN

# 创建结果目录
os.makedirs('visualizations', exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def visualize_model_architecture(model, input_shape, filename):
    """可视化模型架构"""
    x = torch.randn(1, *input_shape)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(filename, directory='visualizations')
    print(f"模型架构图已保存为: visualizations/{filename}.png")

def visualize_feature_maps(model, x_sample, layer_name, filename):
    """可视化指定层的特征图"""
    # 注册钩子来获取特征图
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 选择一个层来可视化
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        if isinstance(layer, nn.Sequential):
            # 如果是Sequential，选择第一个卷积层
            for i, module in enumerate(layer):
                if isinstance(module, nn.Conv1d):
                    module.register_forward_hook(get_activation(f'{layer_name}_{i}'))
        else:
            layer.register_forward_hook(get_activation(layer_name))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(x_sample).unsqueeze(0)  # 添加批次维度
        model(x)
    
    # 可视化激活
    for name, feat in activation.items():
        fig, axs = plt.subplots(min(8, feat.size(1)), 1, figsize=(10, 2*min(8, feat.size(1))))
        
        # 如果只有一个通道
        if feat.size(1) == 1:
            plt.plot(feat[0, 0, :].cpu().numpy())
            plt.title(f'特征图: {name}')
        else:
            for i in range(min(8, feat.size(1))):
                axs[i].plot(feat[0, i, :].cpu().numpy())
                axs[i].set_title(f'通道 {i}')
            fig.suptitle(f'特征图: {name}')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{filename}_{name}.png')
        plt.close()
    
    print(f"特征图已保存到 visualizations/ 目录")

def visualize_filters(model, layer_name, filename):
    """可视化卷积滤波器"""
    # 获取指定层
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        
        # 如果是Sequential，查找第一个卷积层
        if isinstance(layer, nn.Sequential):
            for i, module in enumerate(layer):
                if isinstance(module, nn.Conv1d):
                    weights = module.weight.data.cpu().numpy()
                    visualize_weights(weights, f'{filename}_{layer_name}_{i}')
        # 如果直接是卷积层
        elif isinstance(layer, nn.Conv1d):
            weights = layer.weight.data.cpu().numpy()
            visualize_weights(weights, f'{filename}_{layer_name}')

def visualize_weights(weights, filename):
    """可视化权重"""
    # weights形状: [out_channels, in_channels, kernel_size]
    n_filters = min(16, weights.shape[0])
    n_channels = weights.shape[1]
    
    fig, axs = plt.subplots(n_filters, n_channels, figsize=(n_channels*2, n_filters*2))
    
    # 调整单个过滤器的情况
    if n_filters == 1 and n_channels == 1:
        axs.plot(weights[0, 0, :])
        axs.set_title(f'Filter 0, Channel 0')
    elif n_filters == 1:
        for c in range(n_channels):
            axs[c].plot(weights[0, c, :])
            axs[c].set_title(f'Filter 0, Channel {c}')
    elif n_channels == 1:
        for f in range(n_filters):
            axs[f].plot(weights[f, 0, :])
            axs[f].set_title(f'Filter {f}, Channel 0')
    else:
        for f in range(n_filters):
            for c in range(n_channels):
                axs[f, c].plot(weights[f, c, :])
                axs[f, c].set_title(f'F{f}, Ch{c}')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}_filters.png')
    plt.close()
    print(f"滤波器权重已保存为: visualizations/{filename}_filters.png")

def visualize_predictions_2d(model, X, y, function_name, model_name):
    """在2D输入空间上可视化预测"""
    if X.shape[1] != 2:
        print("此可视化仅适用于2D输入数据")
        return
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 合并为特征数组
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # 预测
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid)
        z = model(grid_tensor).cpu().numpy().reshape(xx.shape)
    
    # 绘制等高线图
    plt.figure(figsize=(10, 8))
    
    # 等高线
    contour = plt.contourf(xx, yy, z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='预测值')
    
    # 实际数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', 
                         edgecolor='k', s=40, alpha=0.7)
    plt.colorbar(scatter, label='实际值')
    
    plt.title(f'{model_name} 在 {function_name} 数据上的预测')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tight_layout()
    plt.savefig(f'visualizations/{function_name}_{model_name}_prediction_map.png')
    plt.close()
    print(f"预测可视化已保存为: visualizations/{function_name}_{model_name}_prediction_map.png")

def main():
    print("===== 开始模型可视化 =====")
    
    # 加载数据
    data_dict = load_data()
    if data_dict is None:
        return
    
    for function_name in ['ackley', 'rosenbrock']:
        x_train, y_train, x_test, y_test = data_dict[function_name]
        input_dim = x_train.shape[1]
        
        # 加载基础CNN模型
        try:
            basic_model = CNN(input_dim)
            basic_model.load_state_dict(torch.load(f'models/cnn_{function_name}.pth'))
            basic_model.eval()
            print(f"成功加载基础CNN模型: cnn_{function_name}.pth")
            
            # 可视化模型架构
            visualize_model_architecture(basic_model, (input_dim,), f'basic_cnn_{function_name}_architecture')
            
            # 可视化卷积层
            visualize_filters(basic_model, 'conv_layers', f'basic_cnn_{function_name}')
            
            # 可视化特征图
            visualize_feature_maps(basic_model, x_test[0], 'conv_layers', f'basic_cnn_{function_name}')
            
            # 如果是2D数据，可视化预测
            if input_dim == 2:
                visualize_predictions_2d(basic_model, x_test, y_test, function_name, 'basic_cnn')
        except Exception as e:
            print(f"可视化基础CNN模型时出错: {str(e)}")
        
        # 加载高级CNN模型
        try:
            advanced_model = AdvancedCNN(input_dim)
            advanced_model.load_state_dict(torch.load(f'models/advanced_cnn_{function_name}.pth'))
            advanced_model.eval()
            print(f"成功加载高级CNN模型: advanced_cnn_{function_name}.pth")
            
            # 可视化模型架构
            visualize_model_architecture(advanced_model, (input_dim,), f'advanced_cnn_{function_name}_architecture')
            
            # 可视化卷积层
            visualize_filters(advanced_model, 'conv1', f'advanced_cnn_{function_name}')
            
            # 可视化特征图
            visualize_feature_maps(advanced_model, x_test[0], 'conv1', f'advanced_cnn_{function_name}')
            
            # 如果是2D数据，可视化预测
            if input_dim == 2:
                visualize_predictions_2d(advanced_model, x_test, y_test, function_name, 'advanced_cnn')
        except Exception as e:
            print(f"可视化高级CNN模型时出错: {str(e)}")
    
    print("===== 模型可视化完成 =====")

if __name__ == "__main__":
    main() 