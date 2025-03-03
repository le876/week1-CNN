import numpy as np
import matplotlib.pyplot as plt
import os

# 打印当前工作目录
print("当前工作目录:", os.getcwd())
print("目录中的文件:", os.listdir())

try:
    # 加载Ackley数据集
    print("尝试加载Ackley数据集...")
    ackley_x_train = np.load('Ackley_x_train.npy')
    ackley_y_train = np.load('Ackley_y_train.npy')
    ackley_x_test = np.load('Ackley_x_test.npy')
    ackley_y_test = np.load('Ackley_y_test.npy')
    
    # 加载Rosenbrock数据集
    print("尝试加载Rosenbrock数据集...")
    rosenbrock_x_train = np.load('Rosenbrock_x_train.npy')
    rosenbrock_y_train = np.load('Rosenbrock_y_train.npy')
    rosenbrock_x_test = np.load('Rosenbrock_x_test.npy')
    rosenbrock_y_test = np.load('Rosenbrock_y_test.npy')
    
    # 打印数据集形状
    print("Ackley训练集 X形状:", ackley_x_train.shape)
    print("Ackley训练集 Y形状:", ackley_y_train.shape)
    print("Ackley测试集 X形状:", ackley_x_test.shape)
    print("Ackley测试集 Y形状:", ackley_y_test.shape)
    print("\nRosenbrock训练集 X形状:", rosenbrock_x_train.shape)
    print("Rosenbrock训练集 Y形状:", rosenbrock_y_train.shape)
    print("Rosenbrock测试集 X形状:", rosenbrock_x_test.shape)
    print("Rosenbrock测试集 Y形状:", rosenbrock_y_test.shape)
    
    # 查看数据的一些统计信息
    print("\nAckley训练集 X统计信息:")
    print("最小值:", np.min(ackley_x_train))
    print("最大值:", np.max(ackley_x_train))
    print("均值:", np.mean(ackley_x_train))
    print("标准差:", np.std(ackley_x_train))
    
    print("\nAckley训练集 Y统计信息:")
    print("最小值:", np.min(ackley_y_train))
    print("最大值:", np.max(ackley_y_train))
    print("均值:", np.mean(ackley_y_train))
    print("标准差:", np.std(ackley_y_train))
    
    print("\nRosenbrock训练集 X统计信息:")
    print("最小值:", np.min(rosenbrock_x_train))
    print("最大值:", np.max(rosenbrock_x_train))
    print("均值:", np.mean(rosenbrock_x_train))
    print("标准差:", np.std(rosenbrock_x_train))
    
    print("\nRosenbrock训练集 Y统计信息:")
    print("最小值:", np.min(rosenbrock_y_train))
    print("最大值:", np.max(rosenbrock_y_train))
    print("均值:", np.mean(rosenbrock_y_train))
    print("标准差:", np.std(rosenbrock_y_train))
    
    # 如果输入是2D的，可视化一些样本点
    if len(ackley_x_train.shape) > 1 and ackley_x_train.shape[1] == 2:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(ackley_x_train[:, 0], ackley_x_train[:, 1], c=ackley_y_train, cmap='viridis')
        plt.colorbar(label='Y值')
        plt.title('Ackley训练集')
        plt.xlabel('X1')
        plt.ylabel('X2')
        
        plt.subplot(1, 2, 2)
        plt.scatter(rosenbrock_x_train[:, 0], rosenbrock_x_train[:, 1], c=rosenbrock_y_train, cmap='viridis')
        plt.colorbar(label='Y值')
        plt.title('Rosenbrock训练集')
        plt.xlabel('X1')
        plt.ylabel('X2')
        
        plt.tight_layout()
        plt.savefig('data_visualization.png')
        plt.close()
        print("\n已保存数据可视化图像到 'data_visualization.png'")
    else:
        print("\n输入数据不是2D的，跳过可视化")
        
except Exception as e:
    print("发生错误:", str(e)) 