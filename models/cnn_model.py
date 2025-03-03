import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    """
    基础CNN模型 - 用于学习Ackley和Rosenbrock函数
    输入: [batch_size, 1, 4, 5] - 单通道4x5的特征矩阵
    输出: [batch_size, 1] - 标量预测值
    """
    def __init__(self, dropout_rate=0.3):
        super(CNN, self).__init__()
        
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
        
        # 注意力机制
        self.attention = SelfAttention(64)
        
        # 全连接层
        # 计算展平后的特征维度: 64 * 3 * 4 = 768 (池化后的尺寸是3x4)
        self.fc1 = nn.Linear(64 * 3 * 4, 128)
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
        
        # 注意力机制
        out = self.attention(out)
        
        # 打印特征图形状以便调试
        # print(f"特征图形状: {out.shape}")
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

class SelfAttention(nn.Module):
    """自注意力机制，用于突出重要特征"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # 生成查询、键和值
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)  # 注意力矩阵
        attention = F.softmax(energy, dim=-1)  # Softmax按最后一维
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x  # 添加原始信息的残差连接
        return out

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保kernel_size是奇数，以便padding能够保持特征图大小
        assert kernel_size % 2 == 1, "Kernel size must be odd for SpatialAttention"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # 显式初始化权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction_ratio不会导致通道数为0
        reduction = max(1, in_channels // reduction_ratio)
        
        # 使用ModuleList而不是Sequential，以便更好地控制初始化
        self.fc1 = nn.Linear(in_channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, in_channels, bias=False)
        
        # 显式初始化权重
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化分支
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.fc1(avg_pool)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        
        # 最大池化分支
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.fc1(max_pool)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)
        
        out = avg_out + max_out
        return torch.sigmoid(out).view(b, c, 1, 1)

class EnhancedCNN(nn.Module):
    """
    增强版CNN模型 - 更深层次的网络结构，多尺度特征提取和增强的注意力机制
    输入: [batch_size, 1, 4, 5] - 单通道4x5的特征矩阵
    输出: [batch_size, 1] - 标量预测值
    """
    def __init__(self, dropout_rate=0.3):
        super(EnhancedCNN, self).__init__()
        
        # 第一卷积块 - 多尺度特征提取
        self.conv1_1x1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.conv1_3x3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)  # 16+32=48
        
        # 第二卷积块
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 简化的残差块
        self.res_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.res_bn = nn.BatchNorm2d(64)
        
        # 第三卷积块 - 增加通道数
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # 简化的注意力机制 - 只使用通道注意力
        self.channel_attention = ChannelAttention(128, reduction_ratio=8)
        
        # 计算全连接层的输入维度
        # 假设输入是[batch_size, 1, 4, 5]
        # 经过卷积和池化后，特征图大小为[batch_size, 128, 3, 4]
        self.fc_input_dim = 128 * 3 * 4
        
        # 简化的全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 多尺度特征提取
        x1 = F.relu(self.conv1_1x1(x))
        x3 = F.relu(self.conv1_3x3(x))
        x = torch.cat([x1, x3], dim=1)
        x = F.relu(self.bn1(x))
        
        # 第二卷积块
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 简化的残差块
        residual = x
        out = F.relu(self.res_bn(self.res_conv(x)))
        out += residual  # 残差连接
        out = F.relu(out)
        
        # 第三卷积块
        out = self.pool(F.relu(self.bn3(self.conv3(out))))
        
        # 简化的注意力机制 - 只使用通道注意力
        out = self.channel_attention(out) * out
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = F.relu(self.bn_fc1(self.fc1(out)))
        out = self.dropout1(out)
        
        out = F.relu(self.bn_fc2(self.fc2(out)))
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out

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
        model = CNN(dropout_rate=dropout_rate)
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