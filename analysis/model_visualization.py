import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from models.cnn_model import BasicCNN, AdvancedCNN, EnhancedCNN

def visualize_model_structure(model, input_shape=(1, 1, 4, 5), output_path='results/model_structure.png'):
    """
    可视化模型结构
    Args:
        model: PyTorch模型
        input_shape: 输入张量的形状
        output_path: 输出图片路径
    """
    x = torch.randn(input_shape)
    y = model(x)
    
    # 使用torchviz生成计算图
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    dot.render(output_path, format='png', cleanup=True)
    print(f"模型结构图已保存到: {output_path}")

def visualize_feature_maps(model, layer_name, input_tensor=None, output_path='results/feature_maps'):
    """
    可视化指定层的特征图
    Args:
        model: PyTorch模型
        layer_name: 要可视化的层名称
        input_tensor: 输入张量，如果为None则生成随机输入
        output_path: 输出目录路径
    """
    if input_tensor is None:
        input_tensor = torch.randn(1, 1, 4, 5)
    
    # 注册钩子函数来获取特征图
    features = {}
    def hook_fn(module, input, output):
        features['output'] = output.detach()
    
    # 获取指定层
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        print(f"可用的层名称:")
        for name, _ in model.named_modules():
            print(f"- {name}")
        raise ValueError(f"未找到层: {layer_name}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 移除钩子
    hook.remove()
    
    # 获取特征图
    feature_maps = features['output'].squeeze(0)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 可视化每个通道的特征图
    num_channels = feature_maps.size(0)
    num_cols = min(8, num_channels)
    num_rows = (num_channels + num_cols - 1) // num_cols
    
    plt.figure(figsize=(2*num_cols, 2*num_rows))
    for i in range(num_channels):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(feature_maps[i].numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{layer_name}_feature_maps.png'))
    plt.close()
    print(f"特征图已保存到: {os.path.join(output_path, f'{layer_name}_feature_maps.png')}")

def visualize_training_history(history_file, output_path='results/training_history.png'):
    """
    可视化训练历史
    Args:
        history_file: 包含训练历史的文件路径
        output_path: 输出图片路径
    """
    # 读取训练历史数据
    history = np.load(history_file)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制评估指标曲线
    if 'train_r2' in history:
        ax2.plot(history['train_r2'], label='Training R²')
        ax2.plot(history['val_r2'], label='Validation R²')
        ax2.set_title('Model R²')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"训练历史图已保存到: {output_path}")

def visualize_model_summary(model, input_shape=(1, 1, 4, 5)):
    """
    打印模型的详细信息
    Args:
        model: PyTorch模型
        input_shape: 输入张量的形状
    """
    print("\n模型结构摘要:")
    print("=" * 50)
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n模型统计信息:")
    print("=" * 50)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 打印每层的输出形状
    print("\n每层输出形状:")
    print("=" * 50)
    x = torch.randn(input_shape)
    
    # 使用forward_features方法获取中间层的输出
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    # 为每一层注册钩子
    for name, module in model.named_modules():
        if not any(isinstance(module, t) for t in [nn.Sequential, BasicCNN, AdvancedCNN, EnhancedCNN]):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 打印每层的输出形状
    for name, activation in activations.items():
        print(f"{name}: {tuple(activation.shape)}")
    
    print(f"最终输出: {tuple(output.shape)}")

def visualize_all_feature_maps(model, input_tensor=None, output_path='results/feature_maps'):
    """
    可视化模型中所有关键层的特征图
    Args:
        model: PyTorch模型
        input_tensor: 输入张量，如果为None则生成随机输入
        output_path: 输出目录路径
    """
    if input_tensor is None:
        input_tensor = torch.randn(1, 1, 4, 5)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 收集所有需要可视化的层
    layers_to_visualize = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            layers_to_visualize[name] = module
    
    # 注册钩子函数来获取特征图
    features = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # 为每一层注册钩子
    for name, module in layers_to_visualize.items():
        hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化每一层的特征图
    for name, feature_maps in features.items():
        feature_maps = feature_maps.squeeze(0)
        
        # 创建图形
        num_channels = feature_maps.size(0)
        num_cols = min(8, num_channels)
        num_rows = (num_channels + num_cols - 1) // num_cols
        
        plt.figure(figsize=(2*num_cols, 2*num_rows))
        plt.suptitle(f'Feature Maps - {name}')
        
        for i in range(num_channels):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(feature_maps[i].numpy(), cmap='viridis')
            plt.axis('off')
            plt.title(f'Channel {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{name}_feature_maps.png'))
        plt.close()
        print(f"特征图已保存到: {os.path.join(output_path, f'{name}_feature_maps.png')}")

def visualize_activation_distributions(model, input_tensor=None, output_path='results/activations'):
    """
    可视化每一层激活值的分布
    Args:
        model: PyTorch模型
        input_tensor: 输入张量，如果为None则生成随机输入
        output_path: 输出目录路径
    """
    if input_tensor is None:
        input_tensor = torch.randn(1, 1, 4, 5)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 收集所有层的激活值
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 为每一层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 创建分布图
    plt.figure(figsize=(15, 5*((len(activations)+2)//3)))
    for i, (name, activation) in enumerate(activations.items(), 1):
        plt.subplot((len(activations)+2)//3, 3, i)
        activation_flat = activation.flatten().numpy()
        sns.histplot(activation_flat, bins=50, kde=True)
        plt.title(f'Activation Distribution - {name}')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'activation_distributions.png'))
    plt.close()
    print(f"激活值分布图已保存到: {os.path.join(output_path, 'activation_distributions.png')}")

def main():
    parser = argparse.ArgumentParser(description='模型可视化工具')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='basic', choices=['basic', 'advanced', 'enhanced'], help='模型类型')
    parser.add_argument('--layer', type=str, default='conv1', help='要可视化的层名称')
    parser.add_argument('--history_file', type=str, help='训练历史文件路径')
    parser.add_argument('--output_dir', type=str, default='results/visualization', help='输出目录')
    parser.add_argument('--show_summary', action='store_true', help='是否显示模型摘要信息')
    parser.add_argument('--all_layers', action='store_true', help='是否显示所有层的特征图')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    if args.model_type == 'basic':
        model = BasicCNN()
    elif args.model_type == 'advanced':
        model = AdvancedCNN()
    else:
        model = EnhancedCNN()
    
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"成功加载模型权重: {args.model_path}")
    except Exception as e:
        print(f"加载模型权重失败: {str(e)}")
        return
    
    model.eval()
    
    # 显示模型摘要
    if args.show_summary:
        visualize_model_summary(model)
    
    # 可视化模型结构
    structure_path = os.path.join(args.output_dir, f'{args.model_type}_structure')
    try:
        visualize_model_structure(model, output_path=structure_path)
    except Exception as e:
        print(f"生成模型结构图失败: {str(e)}")
    
    # 可视化所有层的特征图
    if args.all_layers:
        visualize_all_feature_maps(model, output_path=os.path.join(args.output_dir, 'feature_maps'))
        visualize_activation_distributions(model, output_path=os.path.join(args.output_dir, 'activations'))
    else:
        # 可视化单个层的特征图
        feature_maps_dir = os.path.join(args.output_dir, 'feature_maps')
        visualize_feature_maps(model, args.layer, output_path=feature_maps_dir)
    
    # 如果提供了训练历史文件，则可视化训练过程
    if args.history_file:
        try:
            history_path = os.path.join(args.output_dir, f'{args.model_type}_training_history.png')
            visualize_training_history(args.history_file, output_path=history_path)
        except Exception as e:
            print(f"生成训练历史图失败: {str(e)}")

if __name__ == '__main__':
    main() 