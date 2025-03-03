import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

# 创建结果目录
os.makedirs('processed_data', exist_ok=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FunctionDataset(Dataset):
    """自定义数据集类，用于Ackley和Rosenbrock函数"""
    
    def __init__(self, x_data, y_data, reshape_to_2d=True, transform=None):
        """
        初始化数据集
        
        Args:
            x_data: 输入特征数据
            y_data: 目标值数据
            reshape_to_2d: 是否将特征重塑为二维形式(4x5)
            transform: 数据增强转换
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data).view(-1, 1)  # 确保y是列向量
        self.reshape_to_2d = reshape_to_2d
        self.transform = transform
        
    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        
        # 应用可能的数据增强
        if self.transform:
            x = self.transform(x)
            
        # 重塑为4x5的二维形式
        if self.reshape_to_2d:
            x = x.view(1, 4, 5)  # [channels, height, width]
        
        return x, self.y_data[idx]

def load_and_preprocess_data(function_name, standardize=True, log_transform=False, reshape_to_2d=True):
    """
    加载和预处理函数数据
    
    Args:
        function_name: 'Ackley'或'Rosenbrock'
        standardize: 是否标准化特征
        log_transform: 是否对目标值进行对数变换(适用于Rosenbrock)
        reshape_to_2d: 是否将特征重塑为二维形式
        
    Returns:
        训练数据集, 验证数据集, 测试数据集, X标准化器, Y标准化器
    """
    # 加载数据
    data_dir = 'data'
    x_train = np.load(os.path.join(data_dir, f'{function_name}_x_train.npy'))
    y_train = np.load(os.path.join(data_dir, f'{function_name}_y_train.npy'))
    x_test = np.load(os.path.join(data_dir, f'{function_name}_x_test.npy'))
    y_test = np.load(os.path.join(data_dir, f'{function_name}_y_test.npy'))
    
    logger.info(f"加载{function_name}数据：训练集形状={x_train.shape}, 测试集形状={x_test.shape}")
    
    # 特征标准化
    x_scaler = None
    if standardize:
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_test = x_scaler.transform(x_test)
        logger.info(f"特征标准化完成")
    
    # 目标值变换(对于Rosenbrock数据可选择对数变换)
    y_scaler = None
    if log_transform and function_name == 'Rosenbrock':
        # 对Rosenbrock目标值进行对数变换
        logger.info(f"对{function_name}目标值进行对数变换")
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
    
    # 将训练集分割为训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = FunctionDataset(x_train, y_train, reshape_to_2d=reshape_to_2d)
    val_dataset = FunctionDataset(x_val, y_val, reshape_to_2d=reshape_to_2d)
    test_dataset = FunctionDataset(x_test, y_test, reshape_to_2d=reshape_to_2d)
    
    return train_dataset, val_dataset, test_dataset, x_scaler, y_scaler

class DataAugmentation:
    """数据增强类，包含各种数据增强方法"""
    
    @staticmethod
    def add_gaussian_noise(x, mean=0, std=0.01):
        """添加高斯噪声"""
        return x + torch.randn_like(x) * std + mean
    
    @staticmethod
    def random_scaling(x, scale_range=(0.95, 1.05)):
        """随机缩放"""
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        return x * scale
    
    @staticmethod
    def generate_synthetic_samples(x, y, n_samples=100):
        """生成合成样本 - 简单的线性插值"""
        n = len(x)
        samples = []
        
        for _ in range(n_samples):
            # 随机选择两个样本
            i, j = np.random.choice(range(n), 2, replace=False)
            # 随机插值系数
            alpha = np.random.rand()
            # 生成新样本
            new_x = x[i] * alpha + x[j] * (1 - alpha)
            new_y = y[i] * alpha + y[j] * (1 - alpha)
            samples.append((new_x, new_y))
            
        new_x = torch.stack([s[0] for s in samples])
        new_y = torch.stack([s[1] for s in samples])
        return new_x, new_y

def get_augmented_dataset(original_dataset, augmentation_factor=0.5):
    """
    创建增强数据集
    
    Args:
        original_dataset: 原始数据集
        augmentation_factor: 增强因子(相对于原始数据集的比例)
        
    Returns:
        增强后的数据集
    """
    n_orig = len(original_dataset)
    n_aug = int(n_orig * augmentation_factor)
    
    # 提取原始数据
    orig_x = original_dataset.x_data
    orig_y = original_dataset.y_data
    
    # 生成合成样本
    aug_x, aug_y = DataAugmentation.generate_synthetic_samples(orig_x, orig_y, n_samples=n_aug)
    
    # 合并原始数据和增强数据
    combined_x = torch.cat([orig_x, aug_x])
    combined_y = torch.cat([orig_y, aug_y])
    
    # 创建新的数据集
    augmented_dataset = FunctionDataset(
        combined_x.numpy(), 
        combined_y.numpy(),
        reshape_to_2d=original_dataset.reshape_to_2d
    )
    
    logger.info(f"数据增强完成：原始样本数={n_orig}, 增强后样本数={len(augmented_dataset)}")
    return augmented_dataset

def visualize_preprocessing(function_name, x_orig, x_processed, y_orig, y_processed):
    """可视化预处理前后的数据分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 特征分布变化(只展示前三个特征)
    for i in range(3):
        axes[0, 0].hist(x_orig[:, i], alpha=0.5, bins=30, label=f'特征 {i+1}')
    axes[0, 0].set_title('原始特征分布')
    axes[0, 0].legend()
    
    for i in range(3):
        axes[0, 1].hist(x_processed[:, i], alpha=0.5, bins=30, label=f'特征 {i+1}')
    axes[0, 1].set_title('预处理后特征分布')
    axes[0, 1].legend()
    
    # 目标值分布变化
    axes[1, 0].hist(y_orig, bins=30)
    axes[1, 0].set_title('原始目标值分布')
    
    axes[1, 1].hist(y_processed, bins=30)
    axes[1, 1].set_title('预处理后目标值分布')
    
    plt.tight_layout()
    plt.savefig(f'preprocessing_visualization_{function_name}.png')
    plt.close()
    
    logger.info(f"预处理可视化已保存为 preprocessing_visualization_{function_name}.png")

def main():
    print("===== 数据处理开始 =====")
    
    # 测试数据预处理流程
    for function_name in ['Ackley', 'Rosenbrock']:
        # Rosenbrock使用对数变换，Ackley不使用
        log_transform = (function_name == 'Rosenbrock')
        
        # 加载和预处理数据
        train_dataset, val_dataset, test_dataset, x_scaler, y_scaler = load_and_preprocess_data(
            function_name, 
            standardize=True,
            log_transform=log_transform,
            reshape_to_2d=True
        )
        
        # 创建临时数据加载器用于测试
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 输出数据形状
        for x_batch, y_batch in train_loader:
            logger.info(f"{function_name} 训练数据批次形状: x={x_batch.shape}, y={y_batch.shape}")
            break
            
        # 可视化预处理前后的数据
        x_orig = np.load(f'data/{function_name}_x_train.npy')
        y_orig = np.load(f'data/{function_name}_y_train.npy')
        
        # 获取预处理后的数据(从加载器中提取一批)
        processed_samples = []
        processed_targets = []
        for x, y in train_loader:
            processed_samples.append(x.view(x.size(0), -1).numpy())  # 将2D形状展平以便可视化
            processed_targets.append(y.numpy())
        
        x_processed = np.vstack(processed_samples)
        y_processed = np.vstack(processed_targets).flatten()
        
        # 可视化
        visualize_preprocessing(function_name, x_orig, x_processed, y_orig, y_processed)
        
        logger.info(f"{function_name} 数据预处理测试完成")
    
    print("===== 数据处理完成 =====")

if __name__ == "__main__":
    main() 