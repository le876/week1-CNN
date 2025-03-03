import os
import sys
import argparse
import logging
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import load_and_preprocess_data
from models.cnn_model import create_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_on_test(model_path, model_type, function, device='cpu'):
    """
    加载模型并在测试集上评估
    
    Args:
        model_path: 模型路径
        model_type: 模型类型 (basic, enhanced, advanced)
        function: 函数类型 (Rosenbrock, Ackley)
        device: 设备 (cpu, cuda)
        
    Returns:
        dict: 包含评估指标的字典
    """
    # 加载数据
    logger.info(f"加载{function}数据集...")
    train_dataset, val_dataset, test_dataset, _, _ = load_and_preprocess_data(
        function_name=function,
        log_transform=False  # 使用MSE时不需要对数变换
    )
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False
    )
    
    # 创建模型
    logger.info(f"创建{model_type}模型...")
    model = create_model(model_type=model_type, dropout_rate=0.3)
    
    # 加载模型权重
    logger.info(f"从{model_path}加载模型权重...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_predictions.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # 处理预测和目标值
    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()
    
    # 计算评估指标
    pearson_corr, p_value = pearsonr(test_targets, test_predictions)
    mse = mean_squared_error(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    
    # 计算目标值的范围，用于归一化
    target_min = np.min(test_targets)
    target_max = np.max(test_targets)
    target_range = target_max - target_min
    
    # 计算归一化的MSE（对所有函数都计算）
    normalized_mse = mse
    if target_range > 0:
        normalized_mse = mse / (target_range ** 2)
    
    # 显示目标值范围和归一化MSE（对所有函数都显示）
    logger.info(f"{function}目标值范围: {target_min:.2f} - {target_max:.2f}")
    logger.info(f"归一化MSE: {normalized_mse:.6f}")
    
    # 打印结果
    logger.info(f"测试集评估结果:")
    logger.info(f"Pearson相关系数: {pearson_corr:.4f} (p值: {p_value:.4e})")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    # 返回结果
    return {
        'pearson': pearson_corr,
        'p_value': p_value,
        'mse': mse,
        'normalized_mse': normalized_mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'target_min': target_min,
        'target_max': target_max
    }

def get_training_config(function):
    """获取训练配置信息"""
    # 根据模型类型和函数设置默认配置
    config = {
        'function': function,
        'data_augmentation': False,
        'batch_size': 64,
        'model_type': 'basic',
        'dropout_rate': 0.3,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'optimizer': 'adam',
        'loss_type': 'mse',
        'scheduler': None,
        'step_size': 30,
        'gamma': 0.1,
        'lr_patience': 10,
        'early_stopping': True,
        'patience': 50,
        'seed': 42,
        'output_dir': 'results',
        'log_interval': 5,
        'no_cuda': False,
        'gamma_focal': 2.0,
        'beta': 0.6,
        'adaptation_rate': 0.01
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='评估已训练模型在测试集上的性能')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, required=True, choices=['basic', 'enhanced', 'advanced'], help='模型类型')
    parser.add_argument('--function', type=str, required=True, choices=['Rosenbrock', 'Ackley'], help='函数类型')
    
    args = parser.parse_args()
    
    # 评估模型
    results = evaluate_model_on_test(
        model_path=args.model_path,
        model_type=args.model_type,
        function=args.function
    )
    
    # 获取训练配置
    config = get_training_config(args.function)
    
    # 保存结果到文件
    output_dir = os.path.dirname(args.model_path)
    result_file = os.path.join(output_dir, f"{args.function}_{args.model_type}_test_results.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("测试集评估结果:\n")
        f.write(f"皮尔逊相关系数: {results['pearson']:.4f}\n")
        
        # 对所有函数都显示目标值范围和归一化MSE
        f.write(f"目标值范围: {results['target_min']:.2f} - {results['target_max']:.2f}\n")
        f.write(f"原始MSE: {results['mse']:.4f}\n")
        f.write(f"归一化MSE: {results['normalized_mse']:.6f}\n")
            
        f.write(f"RMSE: {results['rmse']:.4f}\n")
        f.write(f"MAE: {results['mae']:.4f}\n")
        f.write(f"R^2: {results['r2']:.4f}\n")
        
        # 使用适当的Loss值
        f.write(f"Loss(MSE): {results['mse']:.4f}\n")
        f.write(f"Loss(归一化MSE): {results['normalized_mse']:.6f}\n\n")
        
        f.write("训练配置:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # 在控制台也显示完整的评估结果
    logger.info(f"目标值范围: {results['target_min']:.2f} - {results['target_max']:.2f}")
    logger.info(f"原始MSE: {results['mse']:.4f}")
    logger.info(f"归一化MSE: {results['normalized_mse']:.6f}")
    
    logger.info(f"结果已保存到 {result_file}")

if __name__ == "__main__":
    main()