import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicCNN(nn.Module):
    """
    基础CNN模型 - 用于学习Ackley和Rosenbrock函数
    输入: [batch_size, 1, 4, 5] - 单通道4x5的特征矩阵
    输出: [batch_size, 1] - 标量预测值
    """
    def __init__(self, dropout_rate=0.3):
        super(BasicCNN, self).__init__()
        
        # 第一卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # 由于输入尺寸较小，使用stride=1
        
        # 第二卷积块
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 全连接层
        # 计算展平后的特征维度: 32 * 4 * 5 = 640
        self.fc1 = nn.Linear(32 * 4 * 5, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二卷积块
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AdvancedCNN(nn.Module):
    """
    进阶CNN模型 - 带有残差连接和更多层
    输入: [batch_size, 1, 4, 5] - 单通道4x5的特征矩阵
    输出: [batch_size, 1] - 标量预测值
    """
    def __init__(self, dropout_rate=0.3):
        super(AdvancedCNN, self).__init__()
        
        # 第一卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 残差块1
        self.res_conv1a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.res_bn1a = nn.BatchNorm2d(32)
        self.res_conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.res_bn1b = nn.BatchNorm2d(32)
        
        # 第二卷积块
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)  # 由于输入尺寸较小，使用stride=1
        
        # 残差块2
        self.res_conv2a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.res_bn2a = nn.BatchNorm2d(64)
        self.res_conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.res_bn2b = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 5, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 残差块1
        residual = x
        out = F.relu(self.res_bn1a(self.res_conv1a(x)))
        out = self.res_bn1b(self.res_conv1b(out))
        out += residual  # 残差连接
        out = F.relu(out)
        
        # 第二卷积块
        out = self.pool(F.relu(self.bn2(self.conv2(out))))
        
        # 残差块2
        residual = out
        out = F.relu(self.res_bn2a(self.res_conv2a(out)))
        out = self.res_bn2b(self.res_conv2b(out))
        out += residual  # 残差连接
        out = F.relu(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

class EnhancedCNN(nn.Module):
    """
    增强版CNN模型 - 更深层次的网络结构，多尺度特征提取和增强的注意力机制
    输入: [batch_size, 1, 4, 5] - 单通道4x5的特征矩阵
    输出: [batch_size, 1] - 标量预测值
    """
    def __init__(self, dropout_rate=0.3):
        super(EnhancedCNN, self).__init__()
        
        # 第一卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二卷积块
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三卷积块
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 5, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def create_model(model_type='basic', dropout_rate=0.3):
    """
    创建模型实例
    
    Args:
        model_type: 'basic', 'advanced'或'enhanced'
        dropout_rate: Dropout比率
        
    Returns:
        模型实例
    """
    if model_type == 'basic':
        model = BasicCNN(dropout_rate=dropout_rate)
        logger.info("创建基础CNN模型")
    elif model_type == 'advanced':
        model = AdvancedCNN(dropout_rate=dropout_rate)
        logger.info("创建进阶CNN模型(带残差连接和注意力机制)")
    elif model_type == 'enhanced':
        model = EnhancedCNN(dropout_rate=dropout_rate)
        logger.info("创建增强CNN模型(多尺度特征提取和增强的注意力机制)")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 打印模型结构
    logger.info(f"模型结构:\n{model}")
    
    # 计算模型参数数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型可训练参数数量: {params}")
    
    return model

if __name__ == "__main__":
    # 测试模型
    for model_type in ['basic', 'advanced', 'enhanced']:
        model = create_model(model_type=model_type)
        
        # 创建随机输入测试
        x = torch.randn(8, 1, 4, 5)  # 批量大小为8，1通道，4x5特征矩阵
        y = model(x)
        
        logger.info(f"{model_type}模型输入形状: {x.shape}, 输出形状: {y.shape}") 