import numpy as np
import os

print("当前工作目录:", os.getcwd())
print("目录中的文件:", os.listdir())

try:
    # 尝试加载一个文件
    file_path = 'Ackley_x_train.npy'
    print(f"尝试加载文件: {file_path}")
    
    if os.path.exists(file_path):
        print(f"文件存在: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        print(f"数据类型: {type(data)}")
        print(f"数据形状: {data.shape if hasattr(data, 'shape') else '无形状属性'}")
        print(f"数据样本: {data[:2] if hasattr(data, '__getitem__') else '无法显示样本'}")
    else:
        print(f"文件不存在: {file_path}")
        
except Exception as e:
    print(f"发生错误: {str(e)}") 