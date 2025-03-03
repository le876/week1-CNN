import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from models.advanced_cnn_model import AdvancedCNN

# 尝试导入基础CNN模型
try:
    from models.cnn_model import CNN
except ImportError:
    print("警告: 无法导入基础CNN模型，可能文件不存在")
    CNN = None

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def load_model(model_path, model_class, input_dim):
    """加载保存的模型"""
    model = model_class(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, x_test, y_test):
    """评估模型性能"""
    with torch.no_grad():
        # 转换为PyTorch张量
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 获取预测
        predictions = model(x_test_tensor).numpy().flatten()
        targets = y_test_tensor.numpy().flatten()
        
        # 计算皮尔逊相关系数
        pearson_corr, p_value = pearsonr(predictions, targets)
        
        # 计算均方误差
        mse = mean_squared_error(predictions, targets)
        
        return {
            'pearson': pearson_corr,
            'mse': mse,
            'predictions': predictions,
            'targets': targets
        }

def plot_comparison(basic_results, advanced_results, function_name):
    """绘制模型比较图"""
    plt.figure(figsize=(15, 10))
    
    # 基础模型预测vs实际值
    plt.subplot(2, 2, 1)
    plt.scatter(basic_results['targets'], basic_results['predictions'], alpha=0.5)
    plt.plot([min(basic_results['targets']), max(basic_results['targets'])], 
             [min(basic_results['targets']), max(basic_results['targets'])], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'基础CNN: {function_name}\n皮尔逊系数: {basic_results["pearson"]:.4f}, MSE: {basic_results["mse"]:.4f}')
    
    # 高级模型预测vs实际值
    plt.subplot(2, 2, 2)
    plt.scatter(advanced_results['targets'], advanced_results['predictions'], alpha=0.5)
    plt.plot([min(advanced_results['targets']), max(advanced_results['targets'])], 
             [min(advanced_results['targets']), max(advanced_results['targets'])], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'高级CNN: {function_name}\n皮尔逊系数: {advanced_results["pearson"]:.4f}, MSE: {advanced_results["mse"]:.4f}')
    
    # 预测误差直方图 - 基础模型
    plt.subplot(2, 2, 3)
    errors = basic_results['predictions'] - basic_results['targets']
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('基础CNN: 预测误差分布')
    
    # 预测误差直方图 - 高级模型
    plt.subplot(2, 2, 4)
    errors = advanced_results['predictions'] - advanced_results['targets']
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('高级CNN: 预测误差分布')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{function_name}.png')
    plt.close()

def main():
    print("===== 模型比较开始 =====")
    
    # 加载数据
    data_dict = load_data()
    if data_dict is None:
        return
    
    # 创建结果目录
    if not os.path.exists('comparison'):
        os.makedirs('comparison')
    
    # 比较两个函数的模型
    for function_name in ['ackley', 'rosenbrock']:
        print(f"\n===== 比较 {function_name} 函数模型 =====")
        
        # 获取测试数据
        _, _, x_test, y_test = data_dict[function_name]
        input_dim = x_test.shape[1]
        
        # 查找基础模型文件
        basic_model_files = [f for f in os.listdir('.') if f.startswith(f'best_model_') and f.endswith('.pth')]
        if basic_model_files:
            # 选择皮尔逊系数最高的模型
            basic_model_file = sorted(basic_model_files, key=lambda x: float(x.split('_')[-1].split('.pth')[0]), reverse=True)[0]
            print(f"找到基础模型: {basic_model_file}")
            
            # 加载基础模型
            basic_model = load_model(basic_model_file, CNN, input_dim)
            
            # 评估基础模型
            basic_results = evaluate_model(basic_model, x_test, y_test)
            print(f"基础CNN - 皮尔逊系数: {basic_results['pearson']:.4f}, MSE: {basic_results['mse']:.4f}")
        else:
            print("未找到基础模型文件，请先运行 cnn_model.py")
            basic_results = None
        
        # 查找高级模型文件
        advanced_model_path = f'results/advanced_{function_name}_model.pth'
        if os.path.exists(advanced_model_path):
            print(f"找到高级模型: {advanced_model_path}")
            
            # 加载高级模型
            checkpoint = torch.load(advanced_model_path)
            advanced_model = AdvancedCNN(input_dim)
            advanced_model.load_state_dict(checkpoint['model_state_dict'])
            advanced_model.eval()
            
            # 评估高级模型
            advanced_results = evaluate_model(advanced_model, x_test, y_test)
            print(f"高级CNN - 皮尔逊系数: {advanced_results['pearson']:.4f}, MSE: {advanced_results['mse']:.4f}")
        else:
            print("未找到高级模型文件，请先运行 advanced_cnn_model.py")
            advanced_results = None
        
        # 如果两个模型都存在，绘制比较图
        if basic_results and advanced_results:
            plot_comparison(basic_results, advanced_results, function_name)
            print(f"比较图已保存为 model_comparison_{function_name}.png")
            
            # 计算改进百分比
            pearson_improvement = (advanced_results['pearson'] - basic_results['pearson']) / abs(basic_results['pearson']) * 100
            mse_improvement = (basic_results['mse'] - advanced_results['mse']) / basic_results['mse'] * 100
            
            print(f"皮尔逊系数改进: {pearson_improvement:.2f}%")
            print(f"MSE改进: {mse_improvement:.2f}%")
            
            # 判断是否达到目标
            if advanced_results['pearson'] >= 0.85:
                print(f"高级CNN模型达到了目标皮尔逊系数 (>= 0.85)")
            else:
                print(f"高级CNN模型未达到目标皮尔逊系数 (>= 0.85)")
    
    print("\n===== 比较完成 =====")

if __name__ == "__main__":
    main() 