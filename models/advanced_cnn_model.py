import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import os
import sys

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 加载数据
def load_data():
    print("正在加载数据...")
    try:
        # 加载Ackley数据集
        ackley_x_train = np.load('Ackley_x_train.npy')
        ackley_y_train = np.load('Ackley_y_train.npy')
        ackley_x_test = np.load('Ackley_x_test.npy')
        ackley_y_test = np.load('Ackley_y_test.npy')
        
        # 加载Rosenbrock数据集
        rosenbrock_x_train = np.load('Rosenbrock_x_train.npy')
        rosenbrock_y_train = np.load('Rosenbrock_y_train.npy')
        rosenbrock_x_test = np.load('Rosenbrock_x_test.npy')
        rosenbrock_y_test = np.load('Rosenbrock_y_test.npy')
        
        print("数据加载成功！")
        
        # 打印数据形状
        print(f"Ackley训练集: X形状 {ackley_x_train.shape}, Y形状 {ackley_y_train.shape}")
        print(f"Ackley测试集: X形状 {ackley_x_test.shape}, Y形状 {ackley_y_test.shape}")
        print(f"Rosenbrock训练集: X形状 {rosenbrock_x_train.shape}, Y形状 {rosenbrock_y_train.shape}")
        print(f"Rosenbrock测试集: X形状 {rosenbrock_x_test.shape}, Y形状 {rosenbrock_y_test.shape}")
        
        return {
            'ackley': (ackley_x_train, ackley_y_train, ackley_x_test, ackley_y_test),
            'rosenbrock': (rosenbrock_x_train, rosenbrock_y_train, rosenbrock_x_test, rosenbrock_y_test)
        }
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None

# 数据预处理
def preprocess_data(data_dict, function_name):
    x_train, y_train, x_test, y_test = data_dict[function_name]
    
    # 标准化特征
    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    
    # 标准化目标值
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 保存缩放器以便后续使用
    scalers = {
        'x': scaler_x,
        'y': scaler_y
    }
    
    # 转换为PyTorch张量
    x_train_tensor = torch.FloatTensor(x_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).view(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled).view(-1, 1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, scalers

# 定义高级CNN模型
class AdvancedCNN(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedCNN, self).__init__()
        
        self.input_dim = input_dim
        
        # 第一组卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第二组卷积层
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第三组卷积层
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 计算卷积层输出大小
        conv_output_size = (input_dim // 8) * 128
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # 残差连接
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1),
            nn.BatchNorm1d(32)
        )
        
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64)
        )
    
    def forward(self, x):
        # 重塑输入为卷积格式 [batch_size, channels, length]
        x = x.view(-1, 1, self.input_dim)
        
        # 第一组卷积层（带残差连接）
        shortcut = self.shortcut1(x)
        x = self.conv1(x)
        x = x + shortcut[:, :, ::2]  # 调整尺寸以匹配下采样
        
        # 第二组卷积层（带残差连接）
        shortcut = self.shortcut2(x)
        x = self.conv2(x)
        x = x + shortcut[:, :, ::2]  # 调整尺寸以匹配下采样
        
        # 第三组卷积层
        x = self.conv3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

# 训练模型
def train_model(model, train_loader, test_loader, x_test_tensor, y_test_tensor, epochs=200, patience=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    test_losses = []
    pearson_scores = []
    
    best_pearson = 0
    best_model_path = f'advanced_model_best.pth'
    
    # 早停设置
    early_stop_counter = 0
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            # 计算皮尔逊相关系数
            predictions = model(x_test_tensor).numpy().flatten()
            targets = y_test_tensor.numpy().flatten()
            pearson_corr, _ = stats.pearsonr(predictions, targets)
            pearson_scores.append(pearson_corr)
            
            # 更新学习率
            scheduler.step(test_loss)
            
            # 保存最佳模型
            if pearson_corr > best_pearson:
                best_pearson = pearson_corr
                torch.save(model.state_dict(), best_model_path)
                print(f"[保存] 新的最佳模型，皮尔逊系数: {best_pearson:.4f}")
                
            # 早停检查
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"[早停] {patience}个epoch内测试损失没有改善")
                break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}, 测试损失: {test_loss:.4f}, 皮尔逊系数: {pearson_corr:.4f}')
    
    print(f'训练完成! 最佳皮尔逊系数: {best_pearson:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 绘制训练过程
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(pearson_scores, label='皮尔逊系数')
    plt.axhline(y=0.85, color='r', linestyle='--', label='目标 (0.85)')
    plt.xlabel('Epoch')
    plt.ylabel('皮尔逊系数')
    plt.title('皮尔逊相关系数')
    plt.legend()
    
    # 预测vs实际值散点图
    plt.subplot(1, 3, 3)
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor).numpy().flatten()
        targets = y_test_tensor.numpy().flatten()
    
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'预测 vs 实际 (皮尔逊系数: {best_pearson:.4f})')
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png')
    plt.close()
    
    return best_pearson, train_losses, test_losses, pearson_scores, model

# 主函数
def main():
    print("===== 高级CNN模型训练开始 =====")
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 加载数据
    data_dict = load_data()
    if data_dict is None:
        return
    
    results = {}
    
    # 训练两个函数
    for function_name in ['ackley', 'rosenbrock']:
        print(f"\n===== 训练 {function_name} 函数 =====")
        
        # 预处理数据
        train_loader, test_loader, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, scalers = preprocess_data(data_dict, function_name)
        
        # 创建模型
        input_dim = data_dict[function_name][0].shape[1]
        model = AdvancedCNN(input_dim)
        print(f"模型创建成功，输入维度: {input_dim}")
        
        # 打印模型结构
        print(model)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总数: {total_params:,}")
        
        # 训练模型
        best_pearson, train_losses, test_losses, pearson_scores, trained_model = train_model(
            model, train_loader, test_loader, x_test_tensor, y_test_tensor, epochs=300, patience=30
        )
        
        # 保存结果
        results[function_name] = {
            'best_pearson': best_pearson,
            'model': trained_model,
            'scalers': scalers
        }
        
        # 保存模型和缩放器
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'scaler_x': scalers['x'],
            'scaler_y': scalers['y'],
            'best_pearson': best_pearson
        }, f'results/advanced_{function_name}_model.pth')
        
        # 结果
        print(f"\n{function_name} 训练完成！最佳皮尔逊系数: {best_pearson:.4f}")
        if best_pearson >= 0.85:
            print(f"恭喜！{function_name} 达到了目标皮尔逊系数 (>= 0.85)")
        else:
            print(f"{function_name} 未达到目标皮尔逊系数 (>= 0.85)")
    
    # 总结
    print("\n===== 训练总结 =====")
    for function_name, result in results.items():
        print(f"{function_name}: 最佳皮尔逊系数 = {result['best_pearson']:.4f} {'(达标)' if result['best_pearson'] >= 0.85 else '(未达标)'}")

if __name__ == "__main__":
    main() 