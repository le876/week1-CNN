import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
import multiprocessing
from torch.utils.data import DataLoader
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import load_and_preprocess_data, get_augmented_dataset
from models.cnn_model import create_model
from utils.training import (
    train_model, evaluate_model, EarlyStopping, visualize_training_history,
    visualize_predictions, visualize_error_distribution, CombinedLoss, HuberCorrelationLoss,
    FocalCorrelationLoss, AdaptiveCorrelationLoss
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为 {seed}")

def optimize_cpu_performance():
    """优化CPU性能设置"""
    # 设置线程数为物理核心数
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    
    # 启用PyTorch的内部优化
    torch.set_float32_matmul_precision('high')
    
    # 启用Intel MKL优化 (如果可用)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    # 启用内存钉扎，减少内存复制开销
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 设置较大的批处理大小以提高吞吐量
    # 启用异步数据加载和预取
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    logger.info(f"CPU性能优化已启用: 使用 {num_cores} 个线程")

def train_and_evaluate(args):
    """运行整个训练和评估流程"""
    # 优化CPU性能
    optimize_cpu_performance()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载和预处理数据
    logger.info(f"加载和预处理 {args.function} 数据集")
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = load_and_preprocess_data(
        function_name=args.function,
        log_transform=False  # 默认不使用对数变换，因为我们使用MSE损失
    )
    
    # 应用数据增强（如果启用）
    if args.data_augmentation:
        logger.info("应用数据增强...")
        augmented_dataset = get_augmented_dataset(train_dataset, augmentation_factor=0.5)
        # 创建新的数据加载器
        train_loader = DataLoader(
            augmented_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0,  # 使用单进程数据加载
            pin_memory=True  # 启用内存钉扎
        )
        logger.info(f"数据增强后的训练集大小: {len(augmented_dataset)}")
    else:
        # 不使用数据增强，直接使用原始数据集
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,  # 使用单进程数据加载
            pin_memory=True
        )
    
    # 创建验证集和测试集的数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,  # 验证时可以使用更大的批量
        shuffle=False,
        num_workers=0,  # 使用单进程数据加载
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size * 2,  # 测试时可以使用更大的批量
        shuffle=False,
        num_workers=0,  # 使用单进程数据加载
        pin_memory=True
    )
    
    # 创建模型
    logger.info(f"创建 {args.model_type} CNN模型")
    model = create_model(model_type=args.model_type, dropout_rate=args.dropout_rate)
    
    # 设置设备(CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 选择损失函数 - 默认使用MSE
    criterion = nn.MSELoss()
    logger.info(f"使用损失函数: MSE")
    
    # 选择优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")
    
    logger.info(f"使用优化器: {optimizer.__class__.__name__}, 学习率: {args.learning_rate}")
    
    # 选择学习率调度器
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, 
                                                       patience=args.lr_patience, verbose=True)
    elif args.scheduler == 'warmup_cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.optim.lr_scheduler import LinearLR
        from torch.optim.lr_scheduler import SequentialLR
        
        # 创建预热调度器
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=int(args.num_epochs * 0.1)  # 预热10%的轮次
        )
        
        # 创建余弦退火调度器
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=int(args.num_epochs * 0.9),  # 余下90%的轮次
            eta_min=args.learning_rate * 0.01  # 最小学习率为初始学习率的1%
        )
        
        # 组合两个调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[int(args.num_epochs * 0.1)]  # 在10%的轮次处切换
        )
    else:
        scheduler = None
    
    if scheduler is not None:
        logger.info(f"使用学习率调度器: {scheduler.__class__.__name__}")
    
    # 设置早停
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        logger.info(f"启用早停，耐心值: {args.patience}")
    
    # 训练模型
    logger.info(f"开始训练模型，最大轮次: {args.num_epochs}")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping,
        model_save_path=os.path.join(args.output_dir, f"{args.function}_{args.model_type}_model.pth"),
        log_interval=args.log_interval
    )
    
    # 可视化训练历史
    visualize_training_history(
        history, 
        save_path=os.path.join(args.output_dir, f"{args.function}_{args.model_type}_history.png")
    )
    
    # 加载最佳模型
    best_model_path = os.path.join(args.output_dir, f"{args.function}_{args.model_type}_model.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # 在测试集上评估模型
    logger.info("在测试集上评估模型...")
    test_results = evaluate_model(model, test_loader, criterion, device)
    test_loss = test_results['loss']
    test_pearson = test_results['pearson']
    logger.info(f"测试集结果 - 损失: {test_loss:.6f}, Pearson相关系数: {test_pearson:.6f}")
    
    # 可视化预测结果
    visualize_predictions(
        model, 
        test_loader, 
        device, 
        save_path=os.path.join(args.output_dir, f"{args.function}_{args.model_type}_predictions.png")
    )
    
    # 可视化误差分布
    visualize_error_distribution(
        model, 
        test_loader, 
        device, 
        save_path=os.path.join(args.output_dir, f"{args.function}_{args.model_type}_errors.png")
    )
    
    # 保存结果到文本文件
    results_path = os.path.join(args.output_dir, f"{args.function}_{args.model_type}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"函数: {args.function}\n")
        f.write(f"批量大小: {args.batch_size}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"权重衰减: {args.weight_decay}\n")
        f.write(f"优化器: {args.optimizer}\n")
        f.write(f"损失函数: {args.loss_type}\n")
        f.write(f"数据增强: {args.data_augmentation}\n")
        f.write(f"Dropout率: {args.dropout_rate}\n")
        f.write(f"早停耐心值: {args.patience if args.early_stopping else 'N/A'}\n")
        f.write(f"学习率调度器: {args.scheduler}\n")
        f.write(f"随机种子: {args.seed}\n\n")
        f.write(f"测试集损失: {test_loss:.6f}\n")
        f.write(f"测试集Pearson相关系数: {test_pearson:.6f}\n")
    
    logger.info(f"结果已保存到 {results_path}")
    
    return test_pearson

def main():
    parser = argparse.ArgumentParser(description='CNN模型训练和评估')
    
    # 数据参数
    parser.add_argument('--function', type=str, default='Rosenbrock', choices=['Ackley', 'Rosenbrock'],
                        help='要学习的函数 (默认: Rosenbrock)')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='是否使用数据增强')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批量大小 (默认: 64)')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='basic', choices=['basic', 'advanced', 'enhanced'],
                        help='CNN模型类型 (默认: basic)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout率 (默认: 0.3)')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练轮次 (默认: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='权重衰减 (默认: 1e-6)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'],
                        help='优化器类型 (默认: adam)')
    parser.add_argument('--loss_type', type=str, default='mse', 
                        choices=['mse', 'huber', 'combined', 'huber_correlation', 'focal_correlation', 'adaptive_correlation'],
                        help='损失函数类型 (默认: mse)')
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=['None', 'step', 'cosine', 'reduce_on_plateau', 'warmup_cosine'],
                        help='学习率调度器类型 (默认: None)')
    parser.add_argument('--step_size', type=int, default=30,
                        help='StepLR的步长 (默认: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='学习率衰减因子 (默认: 0.1)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='ReduceLROnPlateau的耐心值 (默认: 10)')
    
    # 早停参数
    parser.add_argument('--early_stopping', action='store_true',
                        help='是否使用早停')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值 (默认: 50)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='日志打印间隔 (默认: 每5个epoch)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA (即使可用)')
    
    # 特殊损失函数参数
    parser.add_argument('--gamma_focal', type=float, default=2.0,
                        help='Focal Loss的gamma参数 (默认: 2.0)')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='组合损失的beta参数 (默认: 0.6)')
    parser.add_argument('--adaptation_rate', type=float, default=0.01,
                        help='自适应损失的适应率 (默认: 0.01)')
    
    args = parser.parse_args()
    
    # 打印所有参数
    logger.info("训练参数:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # 运行训练和评估
    test_pearson = train_and_evaluate(args)
    
    # 打印最终结果
    logger.info(f"最终测试集Pearson相关系数: {test_pearson:.6f}")
    
    # 检查是否达到目标
    if test_pearson > 0.85:
        logger.info("🎉 成功达到目标Pearson相关系数 > 0.85!")
    else:
        logger.info(f"❌ 未达到目标Pearson相关系数 > 0.85. 实际值: {test_pearson:.6f}")

if __name__ == "__main__":
    main() 