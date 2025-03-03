import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from models.cnn_model import CNN, load_data
from models.advanced_cnn_model import AdvancedCNN

# 创建结果目录
os.makedirs('explanations', exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def integrated_gradients_explanation(model, inputs, target_class=0):
    """使用整合梯度方法解释模型预测"""
    model.eval()
    ig = IntegratedGradients(model)
    
    # 转换输入为张量
    input_tensor = torch.FloatTensor(inputs)
    
    # 计算特征归因
    attributions, delta = ig.attribute(input_tensor, target=target_class, return_convergence_delta=True)
    
    return attributions.detach().numpy(), delta.detach().numpy()

def occlusion_explanation(model, inputs, target_class=0):
    """使用遮挡方法解释模型预测"""
    model.eval()
    occlusion = Occlusion(model)
    
    # 转换输入为张量
    input_tensor = torch.FloatTensor(inputs)
    
    # 计算特征归因
    # 对于一维输入，窗口大小为特征数量的10%
    window_size = max(1, int(inputs.shape[-1] * 0.1))
    attributions = occlusion.attribute(input_tensor, 
                                     sliding_window_shapes=(window_size,),
                                     target=target_class)
    
    return attributions.detach().numpy()

def layer_gradcam_explanation(model, inputs, layer_name, target_class=0):
    """使用Layer GradCAM方法解释模型预测"""
    model.eval()
    
    # 获取指定层
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        
        # 如果是Sequential，找到第一个卷积层
        if isinstance(layer, torch.nn.Sequential):
            for module in layer:
                if isinstance(module, torch.nn.Conv1d):
                    layer = module
                    break
    else:
        print(f"找不到层: {layer_name}")
        return None
    
    # 创建GradCAM对象
    grad_cam = LayerGradCam(model, layer)
    
    # 转换输入为张量
    input_tensor = torch.FloatTensor(inputs)
    
    # 计算特征归因
    attributions = grad_cam.attribute(input_tensor, target=target_class)
    
    return attributions.detach().numpy()

def plot_feature_attributions(inputs, attributions, title, filename):
    """绘制特征归因图"""
    plt.figure(figsize=(12, 6))
    
    # 如果是单个样本
    if len(inputs.shape) == 1 or inputs.shape[0] == 1:
        if len(inputs.shape) > 1:
            inputs = inputs[0]
            attributions = attributions[0]
        
        # 绘制归因值
        plt.subplot(2, 1, 1)
        plt.bar(range(len(attributions)), attributions)
        plt.title(f'{title} - 特征归因')
        plt.xlabel('特征')
        plt.ylabel('归因值')
        
        # 绘制输入值
        plt.subplot(2, 1, 2)
        plt.plot(inputs)
        plt.title('输入值')
        plt.xlabel('特征')
        plt.ylabel('值')
    else:
        # 如果是多个样本，只绘制前5个
        n_samples = min(5, inputs.shape[0])
        fig, axs = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
        
        for i in range(n_samples):
            # 绘制归因值
            axs[i, 0].bar(range(len(attributions[i])), attributions[i])
            axs[i, 0].set_title(f'样本 {i} - 特征归因')
            
            # 绘制输入值
            axs[i, 1].plot(inputs[i])
            axs[i, 1].set_title(f'样本 {i} - 输入值')
    
    plt.tight_layout()
    plt.savefig(f'explanations/{filename}.png')
    plt.close()
    print(f"特征归因图已保存为: explanations/{filename}.png")

def plot_2d_attributions(inputs, attributions, title, filename):
    """为2D输入绘制特征归因热图"""
    if inputs.shape[1] != 2:
        print("此可视化仅适用于2D输入数据")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 如果是单个样本
    if inputs.shape[0] == 1:
        plt.scatter(inputs[0, 0], inputs[0, 1], c='r', s=100, marker='*')
        plt.title(f'{title} - 单个样本特征归因')
        plt.text(inputs[0, 0], inputs[0, 1], f'X1归因: {attributions[0, 0]:.4f}\nX2归因: {attributions[0, 1]:.4f}')
    else:
        # 创建散点图，颜色表示总归因值（绝对值和）
        total_attr = np.abs(attributions).sum(axis=1)
        scatter = plt.scatter(inputs[:, 0], inputs[:, 1], c=total_attr, cmap='viridis', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, label='总归因值')
        
        # 绘制归因方向向量
        for i in range(min(20, len(inputs))):  # 仅绘制前20个样本的向量
            plt.arrow(inputs[i, 0], inputs[i, 1], 
                     attributions[i, 0]*0.1, attributions[i, 1]*0.1, 
                     head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        plt.title(f'{title} - 特征归因分布')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tight_layout()
    plt.savefig(f'explanations/{filename}.png')
    plt.close()
    print(f"2D特征归因图已保存为: explanations/{filename}.png")

def main():
    print("===== 开始模型解释 =====")
    
    # 加载数据
    data_dict = load_data()
    if data_dict is None:
        return
    
    for function_name in ['ackley', 'rosenbrock']:
        x_train, y_train, x_test, y_test = data_dict[function_name]
        input_dim = x_train.shape[1]
        
        # 选择一些样本进行解释
        samples = x_test[:5]  # 选择前5个测试样本
        
        # 基础CNN模型
        try:
            basic_model = CNN(input_dim)
            basic_model.load_state_dict(torch.load(f'models/cnn_{function_name}.pth'))
            basic_model.eval()
            print(f"成功加载基础CNN模型: cnn_{function_name}.pth")
            
            # 整合梯度解释
            print("计算整合梯度解释...")
            ig_attr, _ = integrated_gradients_explanation(basic_model, samples)
            plot_feature_attributions(samples, ig_attr, f'基础CNN {function_name} - 整合梯度', 
                                     f'basic_cnn_{function_name}_ig')
            
            # 如果是2D输入，绘制特殊的2D归因图
            if input_dim == 2:
                plot_2d_attributions(samples, ig_attr, f'基础CNN {function_name} - 整合梯度', 
                                    f'basic_cnn_{function_name}_ig_2d')
            
            # 遮挡解释
            print("计算遮挡解释...")
            occlusion_attr = occlusion_explanation(basic_model, samples)
            plot_feature_attributions(samples, occlusion_attr, f'基础CNN {function_name} - 遮挡', 
                                     f'basic_cnn_{function_name}_occlusion')
            
            # Layer GradCAM解释
            print("计算Layer GradCAM解释...")
            gradcam_attr = layer_gradcam_explanation(basic_model, samples, 'conv_layers')
            if gradcam_attr is not None:
                plot_feature_attributions(samples, gradcam_attr, f'基础CNN {function_name} - GradCAM', 
                                         f'basic_cnn_{function_name}_gradcam')
        except Exception as e:
            print(f"解释基础CNN模型时出错: {str(e)}")
        
        # 高级CNN模型
        try:
            advanced_model = AdvancedCNN(input_dim)
            advanced_model.load_state_dict(torch.load(f'models/advanced_cnn_{function_name}.pth'))
            advanced_model.eval()
            print(f"成功加载高级CNN模型: advanced_cnn_{function_name}.pth")
            
            # 整合梯度解释
            print("计算整合梯度解释...")
            ig_attr, _ = integrated_gradients_explanation(advanced_model, samples)
            plot_feature_attributions(samples, ig_attr, f'高级CNN {function_name} - 整合梯度', 
                                     f'advanced_cnn_{function_name}_ig')
            
            # 如果是2D输入，绘制特殊的2D归因图
            if input_dim == 2:
                plot_2d_attributions(samples, ig_attr, f'高级CNN {function_name} - 整合梯度', 
                                    f'advanced_cnn_{function_name}_ig_2d')
            
            # 遮挡解释
            print("计算遮挡解释...")
            occlusion_attr = occlusion_explanation(advanced_model, samples)
            plot_feature_attributions(samples, occlusion_attr, f'高级CNN {function_name} - 遮挡', 
                                     f'advanced_cnn_{function_name}_occlusion')
            
            # Layer GradCAM解释
            print("计算Layer GradCAM解释...")
            gradcam_attr = layer_gradcam_explanation(advanced_model, samples, 'conv1')
            if gradcam_attr is not None:
                plot_feature_attributions(samples, gradcam_attr, f'高级CNN {function_name} - GradCAM', 
                                         f'advanced_cnn_{function_name}_gradcam')
        except Exception as e:
            print(f"解释高级CNN模型时出错: {str(e)}")
    
    print("===== 模型解释完成 =====")

if __name__ == "__main__":
    main() 