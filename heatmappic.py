import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib

# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun'] 
matplotlib.rcParams['font.family']='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

# 指定数据集目录
directory = os.getcwd()+'/dataset'  # 更改为实际路径

# 函数：中值滤波
def median_filter(data):
    # 按列应用中值滤波，假设kernel_size为3
    return medfilt(data, kernel_size=(1, 3))

# 函数：最小-最大规范化
def min_max_normalization(data):
    # 按行应用最小-最大规范化
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data.T).T  # 转置是为了按行规范化

# 函数：绘制和保存热力图
def save_heatmap(data, file_name,title):
    plt.figure(figsize=(10, 8))

    # 如果data是DataFrame, 转换为NumPy数组
    data_array = data.values if isinstance(data, pd.DataFrame) else data

    # 检查数组中是否有无穷值
    if np.isinf(data_array).any():
        # 替换无穷大值为NaN, 然后填充为该列的最小值
        data_array = np.where(np.isinf(data_array), np.nan, data_array)
        min_per_column = np.nanmin(data_array, axis=0)
        data_array = np.where(np.isnan(data_array), min_per_column, data_array)

    # 绘制热力图
    sns.heatmap(data_array, cmap="coolwarm", cbar_kws={'label': 'Signal Strength'})
    plt.xlabel("Subcarriers")
    plt.ylabel("Timestamp")
    plt.title(title+"热力图")
    plt.savefig(file_name)
    plt.close()
    print(f"热力图已保存为{file_name}")

# 遍历目录下的所有csv文件
for csv_file in glob.glob(os.path.join(directory, '*.csv')):
    df = pd.read_csv(csv_file)

    # 提取幅度和相位数据
    amplitude_cols = [col for col in df.columns if col.startswith('amplitude_')]
    phase_cols = [col for col in df.columns if col.startswith('phase_')]

    amplitude_data = df[amplitude_cols]
    phase_data = df[phase_cols]

    # 保存原始幅度和相位热力图
    save_heatmap(amplitude_data, csv_file.replace('.csv', '_幅度.png'),title='幅度')
    save_heatmap(phase_data, csv_file.replace('.csv', '_相位.png'),title='相位')

    # 应用中值滤波和规范化
    amplitude_filtered_normalized = min_max_normalization(median_filter(amplitude_data))
    phase_filtered_normalized = min_max_normalization(median_filter(phase_data))

    # # 保存处理后的幅度和相位热力图
    save_heatmap(amplitude_filtered_normalized, csv_file.replace('.csv', '_幅度滤波.png'),title='幅度滤波')
    save_heatmap(phase_filtered_normalized, csv_file.replace('.csv', '_相位滤波.png'),title='相位滤波')

