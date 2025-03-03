import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats
import pandas as pd

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子
np.random.seed(42)

# 创建结果目录
os.makedirs('analysis_results', exist_ok=True)

def load_datasets():
    """加载Ackley和Rosenbrock数据集"""
    print("加载数据集...")
    
    # 数据文件路径
    data_dir = 'data'
    
    try:
        # 加载Ackley数据集
        ackley_x_train = np.load(f'{data_dir}/Ackley_x_train.npy')
        ackley_y_train = np.load(f'{data_dir}/Ackley_y_train.npy')
        ackley_x_test = np.load(f'{data_dir}/Ackley_x_test.npy')
        ackley_y_test = np.load(f'{data_dir}/Ackley_y_test.npy')
        
        # 加载Rosenbrock数据集
        rosenbrock_x_train = np.load(f'{data_dir}/Rosenbrock_x_train.npy')
        rosenbrock_y_train = np.load(f'{data_dir}/Rosenbrock_y_train.npy')
        rosenbrock_x_test = np.load(f'{data_dir}/Rosenbrock_x_test.npy')
        rosenbrock_y_test = np.load(f'{data_dir}/Rosenbrock_y_test.npy')
        
        print("数据加载成功！")
        
        # 数据集信息
        datasets = {
            'ackley': {
                'x_train': ackley_x_train,
                'y_train': ackley_y_train,
                'x_test': ackley_x_test,
                'y_test': ackley_y_test
            },
            'rosenbrock': {
                'x_train': rosenbrock_x_train,
                'y_train': rosenbrock_y_train,
                'x_test': rosenbrock_x_test,
                'y_test': rosenbrock_y_test
            }
        }
        
        # 打印数据集形状
        for name, data in datasets.items():
            print(f"\n{name.capitalize()}数据集:")
            print(f"  训练集: X形状={data['x_train'].shape}, Y形状={data['y_train'].shape}")
            print(f"  测试集: X形状={data['x_test'].shape}, Y形状={data['y_test'].shape}")
        
        return datasets
    
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None

def describe_datasets(datasets):
    """对数据集进行统计描述"""
    print("\n数据集统计分析:")
    
    for name, data in datasets.items():
        print(f"\n{name.capitalize()}数据集统计:")
        
        # 特征统计
        x_train_stats = {
            '均值': np.mean(data['x_train'], axis=0),
            '标准差': np.std(data['x_train'], axis=0),
            '最小值': np.min(data['x_train'], axis=0),
            '最大值': np.max(data['x_train'], axis=0),
            '中位数': np.median(data['x_train'], axis=0)
        }
        
        # 目标值统计
        y_train_stats = {
            '均值': np.mean(data['y_train']),
            '标准差': np.std(data['y_train']),
            '最小值': np.min(data['y_train']),
            '最大值': np.max(data['y_train']),
            '中位数': np.median(data['y_train'])
        }
        
        # 打印特征统计(仅显示前5个特征)
        print("  特征统计 (前5个特征):")
        feature_dims = min(5, data['x_train'].shape[1])
        for stat_name, stat_value in x_train_stats.items():
            print(f"    {stat_name}: {stat_value[:feature_dims]}")
        
        # 打印目标值统计
        print("\n  目标值统计:")
        for stat_name, stat_value in y_train_stats.items():
            print(f"    {stat_name}: {stat_value}")
        
        # 检查异常值和缺失值
        print("\n  数据质量检查:")
        print(f"    特征中的NaN值: {np.isnan(data['x_train']).sum()}")
        print(f"    目标值中的NaN值: {np.isnan(data['y_train']).sum()}")
        print(f"    特征中的Inf值: {np.isinf(data['x_train']).sum()}")
        print(f"    目标值中的Inf值: {np.isinf(data['y_train']).sum()}")
        
        # 创建一个简单的表格来显示完整的统计信息
        stats_df = pd.DataFrame({
            '特征均值': x_train_stats['均值'],
            '特征标准差': x_train_stats['标准差'],
            '特征最小值': x_train_stats['最小值'],
            '特征最大值': x_train_stats['最大值']
        })
        
        # 保存统计信息到CSV文件
        stats_file = f'analysis_results/{name}_statistics.csv'
        stats_df.to_csv(stats_file)
        print(f"\n  详细统计信息已保存到: {stats_file}")

def analyze_ackley_function():
    """分析Ackley函数的数学特性"""
    print("\nAckley函数分析:")
    print("  Ackley函数是一个广泛用于测试优化算法的函数，特点是:")
    print("  - 在整个搜索空间中具有许多局部最小值")
    print("  - 有一个全局最小值位于原点")
    print("  - 函数形状像一个带有中心洞的扁平区域")
    print("  - 随着维度增加，函数会变得更加复杂和难以优化")
    print("  - 数学表达式: f(x) = -20*exp(-0.2*sqrt(0.5*(x1^2+x2^2))) - exp(0.5*(cos(2*pi*x1)+cos(2*pi*x2))) + e + 20")

def analyze_rosenbrock_function():
    """分析Rosenbrock函数的数学特性"""
    print("\nRosenbrock函数分析:")
    print("  Rosenbrock函数(也称为香蕉函数)特点是:")
    print("  - 有一个狭长的抛物线形状的山谷")
    print("  - 全局最小值位于山谷内部的一点")
    print("  - 找到山谷相对容易，但找到山谷中的最小值非常困难")
    print("  - 常用于测试优化算法的收敛性能")
    print("  - 数学表达式: f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]")

def visualize_distributions(datasets):
    """可视化数据分布"""
    print("\n绘制数据分布图...")
    
    for name, data in datasets.items():
        # 特征分布直方图 (对前几个特征)
        feature_dims = min(5, data['x_train'].shape[1])
        plt.figure(figsize=(15, 10))
        
        for i in range(feature_dims):
            plt.subplot(2, 3, i+1)
            sns.histplot(data['x_train'][:, i], kde=True)
            plt.title(f'特征 {i+1} 分布')
            plt.xlabel('值')
            plt.ylabel('频率')
        
        # 目标值分布
        plt.subplot(2, 3, feature_dims+1)
        sns.histplot(data['y_train'], kde=True)
        plt.title('目标值分布')
        plt.xlabel('值')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/{name}_distributions.png')
        plt.close()
        
        print(f"  {name.capitalize()}数据分布图已保存")

def visualize_feature_target_relationships(datasets):
    """可视化特征与目标值的关系"""
    print("\n绘制特征与目标值关系图...")
    
    for name, data in datasets.items():
        # 仅选择前几个特征以避免图表过于复杂
        feature_dims = min(5, data['x_train'].shape[1])
        
        plt.figure(figsize=(15, 10))
        
        for i in range(feature_dims):
            plt.subplot(2, 3, i+1)
            plt.scatter(data['x_train'][:, i], data['y_train'], alpha=0.5)
            
            # 添加趋势线
            z = np.polyfit(data['x_train'][:, i], data['y_train'], 1)
            p = np.poly1d(z)
            plt.plot(data['x_train'][:, i], p(data['x_train'][:, i]), "r--")
            
            # 计算相关系数
            corr, _ = stats.pearsonr(data['x_train'][:, i], data['y_train'])
            
            plt.title(f'特征 {i+1} vs 目标值 (相关系数: {corr:.3f})')
            plt.xlabel(f'特征 {i+1}')
            plt.ylabel('目标值')
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/{name}_feature_target_relationships.png')
        plt.close()
        
        print(f"  {name.capitalize()}特征与目标值关系图已保存")

def visualize_feature_interactions(datasets):
    """可视化特征间的交互关系"""
    print("\n绘制特征交互关系图...")
    
    for name, data in datasets.items():
        # 限制特征数量，避免组合过多
        feature_dims = min(4, data['x_train'].shape[1])
        
        # 如果是二维数据，绘制等高线图
        if data['x_train'].shape[1] == 2:
            plt.figure(figsize=(12, 10))
            
            # 创建网格
            x_min, x_max = data['x_train'][:, 0].min() - 1, data['x_train'][:, 0].max() + 1
            y_min, y_max = data['x_train'][:, 1].min() - 1, data['x_train'][:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # 对每个网格点预测目标值（这里使用一个简单的线性插值）
            from scipy.interpolate import griddata
            grid_z = griddata(data['x_train'][:, :2], data['y_train'], (xx, yy), method='cubic')
            
            # 绘制等高线图
            plt.contourf(xx, yy, grid_z, 50, cmap='viridis', alpha=0.8)
            plt.colorbar(label='目标值')
            
            # 绘制训练点
            plt.scatter(data['x_train'][:, 0], data['x_train'][:, 1], 
                       c=data['y_train'], cmap='plasma', edgecolor='k', s=40)
            
            plt.title(f'{name.capitalize()}函数的二维表示')
            plt.xlabel('特征 1')
            plt.ylabel('特征 2')
            
            plt.tight_layout()
            plt.savefig(f'analysis_results/{name}_contour.png')
            plt.close()
            
            print(f"  {name.capitalize()}函数等高线图已保存")
        
        # 对所有数据，计算特征间的相关性矩阵
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(data['x_train'][:, :feature_dims], rowvar=False)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   xticklabels=[f'特征 {i+1}' for i in range(feature_dims)],
                   yticklabels=[f'特征 {i+1}' for i in range(feature_dims)])
        plt.title(f'{name.capitalize()}数据集特征相关性矩阵')
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/{name}_feature_correlations.png')
        plt.close()
        
        print(f"  {name.capitalize()}特征相关性矩阵已保存")

def calculate_training_test_similarity(datasets):
    """计算训练集和测试集的相似度"""
    print("\n分析训练集和测试集的相似度...")
    
    for name, data in datasets.items():
        # 获取训练集和测试集的特征
        x_train = data['x_train']
        x_test = data['x_test']
        
        # 计算训练集和测试集的特征统计
        train_mean = np.mean(x_train, axis=0)
        train_std = np.std(x_train, axis=0)
        test_mean = np.mean(x_test, axis=0)
        test_std = np.std(x_test, axis=0)
        
        # 计算均值和标准差的相对差异
        mean_diff = np.abs(train_mean - test_mean) / (np.abs(train_mean) + 1e-10)
        std_diff = np.abs(train_std - test_std) / (np.abs(train_std) + 1e-10)
        
        # 打印结果
        print(f"\n  {name.capitalize()}数据集的训练集和测试集相似度:")
        print(f"    特征均值相对差异(平均): {np.mean(mean_diff):.4f}")
        print(f"    特征标准差相对差异(平均): {np.mean(std_diff):.4f}")
        
        # 对于目标值
        y_train = data['y_train']
        y_test = data['y_test']
        
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        y_test_mean = np.mean(y_test)
        y_test_std = np.std(y_test)
        
        y_mean_diff = np.abs(y_train_mean - y_test_mean) / (np.abs(y_train_mean) + 1e-10)
        y_std_diff = np.abs(y_train_std - y_test_std) / (np.abs(y_train_std) + 1e-10)
        
        print(f"    目标值均值相对差异: {y_mean_diff:.4f}")
        print(f"    目标值标准差相对差异: {y_std_diff:.4f}")
        
        # 可视化训练集和测试集的分布比较（对于前几个特征）
        feature_dims = min(3, x_train.shape[1])
        
        plt.figure(figsize=(15, 5 * feature_dims))
        
        for i in range(feature_dims):
            plt.subplot(feature_dims, 1, i+1)
            sns.kdeplot(x_train[:, i], label='训练集')
            sns.kdeplot(x_test[:, i], label='测试集')
            plt.title(f'特征 {i+1} 分布比较')
            plt.xlabel('值')
            plt.ylabel('密度')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/{name}_train_test_comparison.png')
        plt.close()
        
        # 目标值分布比较
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_train, label='训练集')
        sns.kdeplot(y_test, label='测试集')
        plt.title(f'{name.capitalize()}数据集目标值分布比较')
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/{name}_train_test_target_comparison.png')
        plt.close()
        
        print(f"  {name.capitalize()}训练集和测试集比较图已保存")

def summarize_challenges(datasets):
    """总结可能的学习挑战"""
    print("\n总结潜在的学习挑战:")
    
    for name, data in datasets.items():
        print(f"\n  {name.capitalize()}数据集挑战分析:")
        
        # 数据规模
        print(f"    1. 数据规模: 训练样本数={data['x_train'].shape[0]}, 特征维度={data['x_train'].shape[1]}")
        if data['x_train'].shape[0] < 1000:
            print("       - 数据量可能较小，考虑使用数据增强")
        
        # 特征分布
        feature_skews = stats.skew(data['x_train'], axis=0)
        max_skew = np.max(np.abs(feature_skews))
        if max_skew > 1.0:
            print(f"    2. 特征分布偏斜: 最大偏度={max_skew:.3f}")
            print("       - 考虑使用特征变换或标准化处理偏斜分布")
        
        # 目标值分布
        y_skew = stats.skew(data['y_train'])
        if abs(y_skew) > 1.0:
            print(f"    3. 目标值分布偏斜: 偏度={y_skew:.3f}")
            print("       - 考虑对目标值进行变换或标准化")
        
        # 特征与目标值的线性相关性
        feature_dims = min(5, data['x_train'].shape[1])
        max_corr = 0
        for i in range(feature_dims):
            corr, _ = stats.pearsonr(data['x_train'][:, i], data['y_train'])
            max_corr = max(max_corr, abs(corr))
        
        if max_corr < 0.3:
            print(f"    4. 特征与目标的线性相关性低: 最大相关系数={max_corr:.3f}")
            print("       - 暗示非线性关系，CNN可能需要足够的复杂度来捕获")
        
        # 特征间的相关性
        if data['x_train'].shape[1] > 1:
            corr_matrix = np.corrcoef(data['x_train'][:, :feature_dims], rowvar=False)
            np.fill_diagonal(corr_matrix, 0)  # 忽略对角线（自相关）
            max_feature_corr = np.max(np.abs(corr_matrix))
            
            if max_feature_corr > 0.8:
                print(f"    5. 特征间高度相关: 最大特征间相关系数={max_feature_corr:.3f}")
                print("       - 特征可能存在冗余，考虑特征选择或降维")

def main():
    """主函数：执行数据分析流程"""
    print("===== 开始数据理解与分析 =====")
    
    # 加载数据集
    datasets = load_datasets()
    if datasets is None:
        return
    
    # 分析数学函数特性
    analyze_ackley_function()
    analyze_rosenbrock_function()
    
    # 数据集统计描述
    describe_datasets(datasets)
    
    # 可视化数据分布
    visualize_distributions(datasets)
    
    # 可视化特征与目标值关系
    visualize_feature_target_relationships(datasets)
    
    # 可视化特征交互
    visualize_feature_interactions(datasets)
    
    # 分析训练集和测试集的相似度
    calculate_training_test_similarity(datasets)
    
    # 总结学习挑战
    summarize_challenges(datasets)
    
    print("\n===== 数据理解与分析完成 =====")
    print(f"可视化结果和统计信息已保存到 'analysis_results' 目录")

if __name__ == "__main__":
    main() 