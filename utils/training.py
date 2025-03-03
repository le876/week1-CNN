import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib as mpl

# 设置中文字体支持 - 简化版本，避免使用_rebuild方法
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制，监控验证集性能，当性能不再提升时停止训练"""
    def __init__(self, patience=7, min_delta=0, mode='max', verbose=True):
        """
        初始化早停对象
        
        Args:
            patience: 等待改善的轮数
            min_delta: 被视为改善的最小变化
            mode: 'min'表示越小越好(如损失),'max'表示越大越好(如相关系数)
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf if mode == 'min' else -np.inf
        
    def __call__(self, val_score, model, model_path):
        score = -val_score if self.mode == 'min' else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, model_path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, model_path)
            self.counter = 0
            
    def save_checkpoint(self, val_score, model, model_path):
        """保存模型检查点"""
        if self.verbose:
            score_str = f'验证指标 {"降低" if self.mode == "min" else "提升"} ({self.val_score_min:.6f} --> {val_score:.6f})'
            logger.info(f'{score_str}，保存模型到 {model_path}')
        torch.save(model.state_dict(), model_path)
        self.val_score_min = val_score

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler=None, 
                num_epochs=100, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                early_stopping=None,
                model_save_path='models/best_model.pth',
                log_interval=5,
                is_log_transform=False):
    """
    训练模型
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器(可选)
        num_epochs: 训练轮数
        device: 训练设备(CPU/GPU)
        early_stopping: 早停对象(可选)
        model_save_path: 模型保存路径
        log_interval: 日志打印间隔
        is_log_transform: 目标值是否进行了对数变换
        
    Returns:
        训练历史记录(包含loss和metrics)
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 将模型移至指定设备
    model.to(device)
    logger.info(f"模型训练将在 {device} 上进行")
    
    # 准备收集训练历史
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_pearson': [], 
        'val_pearson': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': [],
        'train_r2': [],
        'val_r2': []
    }
    
    best_val_pearson = -1.0
    start_time = time.time()
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失和收集预测
            train_loss += loss.item() * data.size(0)
            train_predictions.extend(output.detach().cpu().numpy())
            train_targets.extend(target.detach().cpu().numpy())
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 处理预测和目标值
        train_predictions = np.array(train_predictions).flatten()
        train_targets = np.array(train_targets).flatten()
        
        # 如果进行了对数变换，需要转换回原始刻度
        if is_log_transform:
            train_predictions = np.expm1(train_predictions)
            train_targets = np.expm1(train_targets)
        
        # 计算训练集评估指标
        train_pearson = pearsonr(train_targets, train_predictions)[0]
        train_mse = mean_squared_error(train_targets, train_predictions)
        train_mae = mean_absolute_error(train_targets, train_predictions)
        train_r2 = r2_score(train_targets, train_predictions)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                output = model(data)
                loss = criterion(output, target)
                
                # 累积损失和收集预测
                val_loss += loss.item() * data.size(0)
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        
        # 处理预测和目标值
        val_predictions = np.array(val_predictions).flatten()
        val_targets = np.array(val_targets).flatten()
        
        # 如果进行了对数变换，需要转换回原始刻度
        if is_log_transform:
            val_predictions = np.expm1(val_predictions)
            val_targets = np.expm1(val_targets)
        
        # 计算验证集评估指标
        val_pearson = pearsonr(val_targets, val_predictions)[0]
        val_mse = mean_squared_error(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_pearson'].append(train_pearson)
        history['val_pearson'].append(val_pearson)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # 打印进度
        if (epoch + 1) % log_interval == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                        f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Pearson: {train_pearson:.4f}, Val Pearson: {val_pearson:.4f}, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型(基于验证集皮尔逊相关系数)
        if val_pearson > best_val_pearson and early_stopping is None:
            best_val_pearson = val_pearson
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Epoch {epoch+1}: 保存最佳模型，验证相关系数: {val_pearson:.4f}')
        
        # 应用早停
        if early_stopping is not None:
            early_stopping(val_pearson, model, model_save_path)
            if early_stopping.early_stop:
                logger.info(f'Early stopping 触发，在 {epoch+1} 轮停止训练')
                break
    
    # 训练结束后的统计
    train_time = time.time() - start_time
    logger.info(f'训练完成，用时 {train_time:.2f} 秒')
    logger.info(f'最佳验证集皮尔逊相关系数: {max(history["val_pearson"]):.4f}')
    
    return history

def evaluate_model(model, 
                  test_loader, 
                  criterion=None, 
                  device='cuda' if torch.cuda.is_available() else 'cpu',
                  is_log_transform=False):
    """
    评估模型
    
    Args:
        model: PyTorch模型
        test_loader: 测试数据加载器 
        criterion: 损失函数(可选)
        device: 评估设备(CPU/GPU)
        is_log_transform: 目标值是否进行了对数变换
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 如果提供了损失函数，计算损失
            if criterion is not None:
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
            
            # 收集预测和真实值
            test_predictions.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # 处理预测和目标值
    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()
    
    # 如果进行了对数变换，需要转换回原始刻度
    if is_log_transform:
        test_predictions = np.expm1(test_predictions)
        test_targets = np.expm1(test_targets)
    
    # 计算评估指标
    test_pearson = pearsonr(test_targets, test_predictions)[0]
    test_mse = mean_squared_error(test_targets, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)
    
    # 如果提供了损失函数，计算平均损失
    if criterion is not None:
        test_loss /= len(test_loader.dataset)
    else:
        test_loss = None
    
    # 打印结果
    logger.info(f'测试集评估结果:')
    logger.info(f'皮尔逊相关系数: {test_pearson:.4f}')
    logger.info(f'MSE: {test_mse:.4f}')
    logger.info(f'RMSE: {test_rmse:.4f}')
    logger.info(f'MAE: {test_mae:.4f}')
    logger.info(f'R²: {test_r2:.4f}')
    if test_loss is not None:
        logger.info(f'Loss: {test_loss:.4f}')
    
    # 返回评估结果
    results = {
        'pearson': test_pearson,
        'mse': test_mse,
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2,
        'loss': test_loss,
        'predictions': test_predictions,
        'targets': test_targets
    }
    
    return results

def visualize_training_history(history, save_path='training_history.png'):
    """
    可视化训练历史
    
    Args:
        history: 包含训练历史的字典
        save_path: 保存路径
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='训练损失')
    ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制皮尔逊相关系数曲线
    ax2.plot(history['train_pearson'], label='训练相关系数')
    ax2.plot(history['val_pearson'], label='验证相关系数')
    ax2.set_title('训练和验证皮尔逊相关系数')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('皮尔逊相关系数')
    ax2.legend()
    ax2.grid(True)
    
    # 保存图像前先调整布局，使用try-except避免警告
    try:
        plt.tight_layout()
    except Exception as e:
        logging.warning(f"tight_layout调整失败: {e}")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    logger.info(f'训练历史图表已保存至 {save_path}')

def visualize_predictions(predictions, targets, title='预测结果', save_path='predictions.png'):
    """
    可视化预测结果与真实值的对比
    
    Args:
        predictions: 模型预测值
        targets: 真实目标值
        title: 图表标题
        save_path: 保存路径
    """
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 计算理想预测线的范围
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    
    # 绘制散点图
    plt.scatter(targets, predictions, alpha=0.6, color='blue', label='预测点')
    
    # 绘制理想预测线 (y=x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
    
    # 计算并显示皮尔逊相关系数
    pearson_corr = pearsonr(targets, predictions)[0]
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # 添加统计信息文本框
    stats_text = (f'统计信息:\n'
                  f'皮尔逊相关系数: {pearson_corr:.4f}\n'
                  f'MSE: {mse:.4f}\n'
                  f'RMSE: {rmse:.4f}')
    plt.annotate(stats_text, 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 verticalalignment='top')
    
    # 添加标题和标签
    plt.title(title)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True)
    plt.legend()
    
    # 保存图像前先调整布局，使用try-except避免警告
    try:
        plt.tight_layout()
    except Exception as e:
        logging.warning(f"tight_layout调整失败: {e}")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    logger.info(f'预测结果图表已保存至 {save_path}')

def visualize_error_distribution(predictions, targets, title='预测误差分布', save_path='error_distribution.png'):
    """
    可视化预测误差分布
    
    Args:
        predictions: 模型预测值
        targets: 真实目标值
        title: 图表标题
        save_path: 保存路径
    """
    # 计算误差
    errors = predictions - targets
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制误差直方图
    n, bins, patches = plt.hist(errors, bins=30, alpha=0.7, color='skyblue', label='误差分布')
    
    # 添加垂直线表示均值和中位数
    plt.axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=1, label='均值')
    plt.axvline(np.median(errors), color='green', linestyle='dashed', linewidth=1, label='中位数')
    
    # 添加统计信息文本框
    stats_text = (f'统计信息:\n'
                  f'均值: {np.mean(errors):.4f}\n'
                  f'标准差: {np.std(errors):.4f}\n'
                  f'中位数: {np.median(errors):.4f}\n'
                  f'最小值: {np.min(errors):.4f}\n'
                  f'最大值: {np.max(errors):.4f}\n'
                  f'中位数: {np.median(errors):.4f}')
    plt.annotate(stats_text, 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 verticalalignment='top')
    
    # 添加标题和标签
    plt.title(title)
    plt.xlabel('预测误差')
    plt.ylabel('频数')
    plt.grid(True)
    plt.legend()
    
    # 保存图像前先调整布局，使用try-except避免警告
    try:
        plt.tight_layout()
    except Exception as e:
        logging.warning(f"tight_layout调整失败: {e}")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    logger.info(f'误差分布图表已保存至 {save_path}')

def get_correlation_loss(y_pred, y_true, eps=1e-8):
    """
    计算相关系数损失函数（1 - 皮尔逊相关系数）
    
    Args:
        y_pred: 预测值 
        y_true: 真实值
        eps: 小值，防止除零错误
        
    Returns:
        相关系数损失
    """
    # 展平张量
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # 计算均值
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    
    # 计算相关系数
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    
    # 返回损失(1 - 相关系数)
    return 1.0 - corr

class CombinedLoss(nn.Module):
    """结合MSE损失和相关系数损失的组合损失函数"""
    def __init__(self, alpha=0.5):
        """
        初始化组合损失函数
        
        Args:
            alpha: MSE损失的权重(0到1之间)，相关系数损失的权重为(1-alpha)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # 计算MSE损失
        mse = self.mse_loss(y_pred, y_true)
        
        # 计算相关系数损失
        corr_loss = get_correlation_loss(y_pred, y_true)
        
        # 组合损失
        loss = self.alpha * mse + (1 - self.alpha) * corr_loss
        
        return loss

class HuberCorrelationLoss(nn.Module):
    """结合Huber损失和相关系数损失的组合损失函数"""
    def __init__(self, delta=1.0, alpha=0.5):
        """
        初始化组合损失函数
        
        Args:
            delta: Huber损失的阈值
            alpha: Huber损失的权重(0到1之间)，相关系数损失的权重为(1-alpha)
        """
        super(HuberCorrelationLoss, self).__init__()
        self.alpha = alpha
        self.huber_loss = nn.HuberLoss(delta=delta)
        
    def forward(self, y_pred, y_true):
        # 计算Huber损失
        huber = self.huber_loss(y_pred, y_true)
        
        # 计算相关系数损失
        corr_loss = get_correlation_loss(y_pred, y_true)
        
        # 组合损失
        loss = self.alpha * huber + (1 - self.alpha) * corr_loss
        
        return loss 

class FocalCorrelationLoss(nn.Module):
    """
    焦点相关系数损失函数 - 对难以优化的样本给予更多关注
    结合MSE、Huber和相关系数损失，并使用动态权重
    """
    def __init__(self, delta=1.0, alpha=0.4, gamma=2.0, beta=0.6):
        """
        初始化焦点相关系数损失函数
        
        Args:
            delta: Huber损失的阈值
            alpha: MSE/Huber损失的初始权重(0到1之间)
            gamma: 焦点损失的聚焦参数，控制对难样本的关注程度
            beta: 相关系数损失的权重系数
        """
        super(FocalCorrelationLoss, self).__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.huber_loss = nn.HuberLoss(delta=delta)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # 计算基础损失
        huber = self.huber_loss(y_pred, y_true)
        mse = self.mse_loss(y_pred, y_true)
        
        # 计算相关系数损失
        corr_loss = get_correlation_loss(y_pred, y_true)
        
        # 计算焦点权重 - 相关系数越低，权重越高
        focal_weight = (corr_loss ** self.gamma)
        
        # 组合损失 - 动态调整权重
        combined_reg_loss = self.alpha * huber + (1 - self.alpha) * mse
        loss = (1 - self.beta) * combined_reg_loss + self.beta * focal_weight * corr_loss
        
        return loss

class AdaptiveCorrelationLoss(nn.Module):
    """
    自适应相关系数损失 - 在训练过程中动态调整损失权重
    """
    def __init__(self, delta=1.0, init_alpha=0.5, adaptation_rate=0.01):
        """
        初始化自适应相关系数损失
        
        Args:
            delta: Huber损失的阈值
            init_alpha: 初始的Huber损失权重
            adaptation_rate: 权重调整速率
        """
        super(AdaptiveCorrelationLoss, self).__init__()
        self.delta = delta
        self.alpha = init_alpha
        self.adaptation_rate = adaptation_rate
        self.huber_loss = nn.HuberLoss(delta=delta)
        self.prev_corr = 0.0
        self.steps = 0
        
    def forward(self, y_pred, y_true):
        # 计算Huber损失
        huber = self.huber_loss(y_pred, y_true)
        
        # 计算相关系数损失
        corr_loss = get_correlation_loss(y_pred, y_true)
        current_corr = 1.0 - corr_loss.item()  # 当前批次的相关系数
        
        # 更新权重 - 如果相关系数提高，增加相关系数损失的权重
        if self.steps > 0:
            corr_change = current_corr - self.prev_corr
            # 如果相关系数提高，增加其权重；否则增加Huber损失权重
            if corr_change > 0:
                self.alpha = max(0.1, self.alpha - self.adaptation_rate)
            else:
                self.alpha = min(0.9, self.alpha + self.adaptation_rate)
        
        self.prev_corr = current_corr
        self.steps += 1
        
        # 组合损失
        loss = self.alpha * huber + (1 - self.alpha) * corr_loss
        
        return loss 