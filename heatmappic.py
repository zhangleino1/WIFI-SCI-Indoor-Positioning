import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util import min_max_normalization, median_filter
import matplotlib
from CSIKit.reader import get_reader
from CSIKit.util import csitools
import re

# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun'] 
matplotlib.rcParams['font.family']='sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

# 指定数据集目录
directory = os.getcwd()+'/dataset'  # 更改为实际路径



# 函数：绘制和保存热力图
def save_heatmap(data, file_name,title):
        # 检查文件是否已存在
    if os.path.exists(file_name):
        print(f"{file_name} 已存在，跳过生成。")
        return
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


if __name__ == '__main__':

    # 遍历目录下的所有.dat文件
    for dat_file in glob.glob(os.path.join(directory, '*.dat')):
        # 从文件名中提取坐标
        match = re.search(r'\((\d+),(\d+)\).dat', os.path.basename(dat_file))
        if match:
            x_label, y_label = match.groups()
            # 读取CSI数据
            my_reader = get_reader(dat_file)
            csi_data = my_reader.read_file(dat_file, scaled=True)

            # 为每个接收天线创建热力图
            for rx_ant in range(3):  # 假设有三个接收天线
                # 获取幅度和相位信息
                csi_amplitude = csitools.get_CSI(csi_data, metric="amplitude")[0][:, :, rx_ant, 0]
                csi_phase = csitools.get_CSI(csi_data, metric="phase")[0][:, :, rx_ant, 0]

                # 创建幅度和相位的热力图
                save_heatmap(csi_amplitude, os.path.join(directory, f"{x_label}_{y_label}_ant{rx_ant}_amplitude_raw.png"), f"Ant{rx_ant} 原始幅度")
                save_heatmap(csi_phase, os.path.join(directory, f"{x_label}_{y_label}_ant{rx_ant}_phase_raw.png"), f"Ant{rx_ant} 原始相位")

                # 应用中值滤波和最小-最大规范化
                amplitude_filtered_normalized = min_max_normalization(median_filter(csi_amplitude))
                phase_filtered_normalized = min_max_normalization(median_filter(csi_phase))

                # 保存处理后的热力图
                save_heatmap(amplitude_filtered_normalized, os.path.join(directory, f"{x_label}_{y_label}_ant{rx_ant}_amplitude_filtered.png"), f"Ant{rx_ant} 滤波后幅度")
                save_heatmap(phase_filtered_normalized, os.path.join(directory, f"{x_label}_{y_label}_ant{rx_ant}_phase_filtered.png"), f"Ant{rx_ant} 滤波后相位")

