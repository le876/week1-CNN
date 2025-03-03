import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import os
import sys
import json
import time
from bayes_opt import BayesianOptimization
from functools import partial

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 创建结果目录
os.makedirs('hyperopt', exist_ok=True)

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
        
        return {
            'ackley': (ackley_x_train, ackley_y_train, ackley_x_test, ackley_y_test),
            'rosenbrock': (rosenbrock_x_train, rosenbrock_y_train, rosenbrock_x_test, rosenbrock_y_test)
        }
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None

# 定义CNN模型
class TunableCNN(nn.Module):
    def __init__(self, input_dim, conv_channels=[32, 64], kernel_size=3, dropout_rate=0.3):
        super(TunableCNN, self).__init__()
        
        self.input_dim = input_dim
        layers = []
        
        in_channels = 1
        current_dim = input_dim
        
        # 创建卷积层
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
            
            # 每两个卷积层后添加一个池化层
            if len(layers) > 3:  # 第一层卷积后不进行池化
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_dim = current_dim // 2
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积输出尺寸
        conv_output_size = current_dim * in_channels
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # 第二层使用较小的dropout
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # 重塑输入为卷积格式 [batch_size, channels, length]
        x = x.view(-1, 1, self.input_dim)
        
        # 通过卷积层
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        
        return x

# 交叉验证训练函数
def cross_validate(model_fn, x_train, y_train, batch_size, lr, weight_decay, n_folds=5, epochs=100, early_stopping=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    x_tensor = torch.FloatTensor(x_train)
    y_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        print(f"Fold {fold+1}/{n_folds}")
        
        # 创建训练和验证数据集
        x_train_fold = x_tensor[train_idx]
        y_train_fold = y_tensor[train_idx]
        x_val_fold = x_tensor[val_idx]
        y_val_fold = y_tensor[val_idx]
        
        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        val_dataset = TensorDataset(x_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_dim = x_train.shape[1]
        model = model_fn(input_dim)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # 评估模式
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            val_loss /= len(val_loader)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"早停于epoch {epoch+1}")
                    break
        
        # 最终验证评分
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_fold).numpy().flatten()
            val_targets = y_val_fold.numpy().flatten()
            r2 = r2_score(val_targets, val_preds)
            mse = np.mean((val_preds - val_targets) ** 2)
        
        print(f"Fold {fold+1} - MSE: {mse:.6f}, R2: {r2:.6f}")
        fold_scores.append(mse)
    
    # 返回平均MSE
    avg_mse = np.mean(fold_scores)
    print(f"平均MSE: {avg_mse:.6f}")
    return avg_mse

# 网格搜索超参数
def grid_search(x_train, y_train, function_name):
    print(f"开始 {function_name} 数据的网格搜索...")
    
    # 定义超参数网格
    param_grid = {
        'batch_size': [16, 32, 64],
        'lr': [0.001, 0.0005, 0.0001],
        'weight_decay': [0, 0.0001, 0.001],
        'conv_channels': [[16, 32], [32, 64], [64, 128]],
        'kernel_size': [3, 5, 7],
        'dropout_rate': [0.2, 0.3, 0.5]
    }
    
    results = []
    best_mse = float('inf')
    best_params = None
    
    # 计算总搜索空间大小
    total_combinations = (
        len(param_grid['batch_size']) *
        len(param_grid['lr']) *
        len(param_grid['weight_decay']) *
        len(param_grid['conv_channels']) *
        len(param_grid['kernel_size']) *
        len(param_grid['dropout_rate'])
    )
    
    print(f"总搜索组合数: {total_combinations}")
    
    # 为了减少搜索空间，我们只搜索部分组合
    # 这里使用一个简化的网格搜索，只变化一个参数
    
    # 基础参数
    base_params = {
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'conv_channels': [32, 64],
        'kernel_size': 3,
        'dropout_rate': 0.3
    }
    
    # 依次改变每个参数
    for param_name in param_grid:
        print(f"\n搜索参数: {param_name}")
        for param_value in param_grid[param_name]:
            # 创建当前参数组合
            current_params = base_params.copy()
            current_params[param_name] = param_value
            
            print(f"评估参数: {param_name}={param_value}, 其他参数保持不变")
            
            # 定义模型创建函数
            def create_model(input_dim):
                return TunableCNN(
                    input_dim=input_dim,
                    conv_channels=current_params['conv_channels'],
                    kernel_size=current_params['kernel_size'],
                    dropout_rate=current_params['dropout_rate']
                )
            
            # 进行交叉验证
            start_time = time.time()
            mse = cross_validate(
                create_model,
                x_train, y_train,
                batch_size=current_params['batch_size'],
                lr=current_params['lr'],
                weight_decay=current_params['weight_decay'],
                n_folds=3,  # 为了速度使用较少的折数
                epochs=50   # 为了速度使用较少的epoch
            )
            duration = time.time() - start_time
            
            # 记录结果
            result = {
                'params': current_params.copy(),
                'mse': mse,
                'duration': duration
            }
            results.append(result)
            
            # 更新最佳参数
            if mse < best_mse:
                best_mse = mse
                best_params = current_params.copy()
                print(f"新的最佳MSE: {best_mse:.6f}, 参数: {best_params}")
    
    # 保存结果
    results_file = f'hyperopt/grid_search_{function_name}_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': {
                'params': best_params,
                'mse': float(best_mse)
            }
        }, f, indent=2)
    
    print(f"网格搜索完成，最佳MSE: {best_mse:.6f}")
    print(f"最佳参数: {best_params}")
    print(f"结果已保存到 {results_file}")
    
    return best_params

# 贝叶斯优化超参数
def bayesian_optimization(x_train, y_train, function_name):
    print(f"开始 {function_name} 数据的贝叶斯优化...")
    
    # 定义贝叶斯优化的目标函数
    def objective(batch_size, learning_rate, weight_decay, dropout_rate, kernel_size, conv1_channels, conv2_channels):
        # 将连续参数转换为离散参数
        batch_size = int(round(batch_size))
        kernel_size = int(round(kernel_size))
        conv1_channels = int(round(conv1_channels))
        conv2_channels = int(round(conv2_channels))
        
        print(f"评估参数: batch_size={batch_size}, lr={learning_rate:.6f}, "
              f"weight_decay={weight_decay:.6f}, dropout={dropout_rate:.2f}, "
              f"kernel_size={kernel_size}, channels=[{conv1_channels}, {conv2_channels}]")
        
        # 定义模型创建函数
        def create_model(input_dim):
            return TunableCNN(
                input_dim=input_dim,
                conv_channels=[conv1_channels, conv2_channels],
                kernel_size=kernel_size,
                dropout_rate=dropout_rate
            )
        
        # 进行交叉验证
        try:
            mse = cross_validate(
                create_model,
                x_train, y_train,
                batch_size=batch_size,
                lr=learning_rate,
                weight_decay=weight_decay,
                n_folds=3,  # 为了速度使用较少的折数
                epochs=50   # 为了速度使用较少的epoch
            )
            
            # 贝叶斯优化是最大化目标，所以我们返回负MSE
            return -mse
        except Exception as e:
            print(f"评估时出错: {str(e)}")
            return -float('inf')  # 返回一个非常低的值
    
    # 定义参数范围
    pbounds = {
        'batch_size': (8, 64),         # 会被四舍五入为整数
        'learning_rate': (1e-4, 1e-2),
        'weight_decay': (1e-6, 1e-3),
        'dropout_rate': (0.1, 0.5),
        'kernel_size': (3, 7),         # 会被四舍五入为整数
        'conv1_channels': (16, 128),   # 会被四舍五入为整数
        'conv2_channels': (32, 256)    # 会被四舍五入为整数
    }
    
    # 创建优化器
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42
    )
    
    # 运行优化
    optimizer.maximize(
        init_points=5,   # 随机探索的初始点数
        n_iter=15        # 贝叶斯优化迭代次数
    )
    
    print("贝叶斯优化完成")
    print(f"最佳参数: {optimizer.max['params']}")
    print(f"最佳MSE: {-optimizer.max['target']:.6f}")
    
    # 保存结果
    results_file = f'hyperopt/bayesian_opt_{function_name}_results.json'
    
    # 将numpy值转换为Python原生类型，以便进行JSON序列化
    best_params = {k: float(v) for k, v in optimizer.max['params'].items()}
    best_target = float(-optimizer.max['target'])
    
    all_results = []
    for i, res in enumerate(optimizer.res):
        params = {k: float(v) for k, v in res['params'].items()}
        all_results.append({
            'iteration': i,
            'params': params,
            'mse': float(-res['target'])
        })
    
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_result': {
                'params': best_params,
                'mse': best_target
            }
        }, f, indent=2)
    
    print(f"结果已保存到 {results_file}")
    
    # 返回最佳参数（转换为正确的类型）
    return {
        'batch_size': int(round(best_params['batch_size'])),
        'lr': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'conv_channels': [int(round(best_params['conv1_channels'])), 
                        int(round(best_params['conv2_channels']))],
        'kernel_size': int(round(best_params['kernel_size'])),
        'dropout_rate': best_params['dropout_rate']
    }

# 使用最佳参数训练模型
def train_best_model(x_train, y_train, x_test, y_test, best_params, function_name):
    print(f"使用最佳参数训练 {function_name} 最终模型...")
    
    # 准备数据
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 创建模型
    input_dim = x_train.shape[1]
    model = TunableCNN(
        input_dim=input_dim,
        conv_channels=best_params['conv_channels'],
        kernel_size=best_params['kernel_size'],
        dropout_rate=best_params['dropout_rate']
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    # 训练循环
    epochs = 200
    early_stopping = 20
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 创建模型目录
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 评估模式
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - 训练MSE: {train_loss:.6f}, 验证MSE: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'models/tuned_cnn_{function_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f"早停于epoch {epoch+1}")
                break
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{function_name} - 训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'hyperopt/tuned_cnn_{function_name}_loss.png')
    plt.close()
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(f'models/tuned_cnn_{function_name}.pth'))
    model.eval()
    
    # 评估性能
    with torch.no_grad():
        test_preds = model(x_test_tensor).numpy().flatten()
        test_targets = y_test_tensor.numpy().flatten()
        r2 = r2_score(test_targets, test_preds)
        mse = np.mean((test_preds - test_targets) ** 2)
    
    print(f"最终测试集 MSE: {mse:.6f}, R2: {r2:.6f}")
    
    # 绘制预测vs实际值图
    plt.figure(figsize=(10, 8))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], 
             [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'调优CNN: {function_name}\nMSE: {mse:.4f}, R2: {r2:.4f}')
    plt.tight_layout()
    plt.savefig(f'hyperopt/tuned_cnn_{function_name}_predictions.png')
    plt.close()
    
    # 保存评估结果
    evaluation = {
        'mse': float(mse),
        'r2': float(r2),
        'best_params': best_params
    }
    
    with open(f'hyperopt/tuned_cnn_{function_name}_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    return model, mse, r2

def main():
    print("===== 超参数调优开始 =====")
    
    # 加载数据
    data_dict = load_data()
    if data_dict is None:
        return
    
    # 对每个函数进行超参数优化
    for function_name in ['ackley', 'rosenbrock']:
        x_train, y_train, x_test, y_test = data_dict[function_name]
        
        print(f"\n{'-'*50}")
        print(f"处理 {function_name} 函数")
        print(f"{'-'*50}\n")
        
        # 执行网格搜索
        grid_best_params = grid_search(x_train, y_train, function_name)
        
        # 执行贝叶斯优化
        bayes_best_params = bayesian_optimization(x_train, y_train, function_name)
        
        # 使用贝叶斯优化的参数训练最终模型
        train_best_model(x_train, y_train, x_test, y_test, bayes_best_params, function_name)
    
    print("===== 超参数调优完成 =====")

if __name__ == "__main__":
    main() 