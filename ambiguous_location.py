import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import torch
from csi_dataset import CSIDataModule
import os
from util import min_max_normalization, median_filter
# 假设CSIDataset类和CSIDataModule类已经在上文给出，并正确导入

# 初始化数据模块
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import glob
import matplotlib.pyplot as plt
import re
class CSIDataHandler:
    def __init__(self, data_directory, time_steps=30):
        self.directory = data_directory
        self.time_steps = time_steps  # 设置时间步长

    def load_data_by_location(self):
        data_by_location = {}
        # 获取所有文件路径
        all_files = glob.glob(os.path.join(self.directory, 'antenna_*_*.csv'))
        # 通过正则表达式组织文件按位置分组
        file_pattern = re.compile(r'antenna_(\d+)_(\d+)_(\d+).csv')
        files_by_location = {}
        for file_path in all_files:
            match = file_pattern.search(os.path.basename(file_path))
            if match:
                ant, x, y = match.groups()
                location_key = (x, y)
                if location_key not in files_by_location:
                    files_by_location[location_key] = []
                files_by_location[location_key].append(file_path)

        # 确保每个位置有三个天线的数据才进行处理
        for location_key, files in files_by_location.items():
            if len(files) == 3:
                all_data = []
                for file_path in sorted(files):  # 根据文件名排序，以保证天线顺序
                    df = pd.read_csv(file_path)
                    amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
                    phase_data = df.filter(regex='^phase_').values.astype(np.float32)
                    
                    # 对整个CSV应用滤波和归一化
                    amplitude_data = min_max_normalization(median_filter(amplitude_data))
                    phase_data = min_max_normalization(median_filter(phase_data))
                    num_records = len(df)
                    # 按时间步长划分数据
                    for start in range(0, num_records, self.time_steps):  
                        end = start + self.time_steps
                        if end <= num_records:
                            slice_amplitude = amplitude_data[start:end, :]
                            slice_phase = phase_data[start:end, :]
                            combined_data = np.stack([slice_amplitude, slice_phase], axis=0)
                            all_data.append(combined_data)
                if all_data:
                    # 将三个天线的数据沿着时间步（第二维）连接起来
                    location_data = np.concatenate(all_data, axis=0)
                    data_by_location[location_key] = location_data
        return data_by_location

    def calculate_correlation(self, data_by_location):
        correlation_results = {}
        for location, data in data_by_location.items():
            coeffs = []
            for i in range(data.shape[1] - 1):  # 时间维度是第二个维度
                for j in range(i + 1, data.shape[1]):
                    xi = data[:, i, :].reshape(-1)
                    xj = data[:, j, :].reshape(-1)
                    coeff, _ = pearsonr(xi, xj)
                    coeffs.append(coeff)
            correlation_results[location] = np.mean(coeffs)
        return correlation_results
# 使用
data_directory = data_dir=os.getcwd()+"/dataset"
handler = CSIDataHandler(data_directory)
data_by_location = handler.load_data_by_location()
correlation_results = handler.calculate_correlation(data_by_location)

# 准备数据
locations = sorted(correlation_results.keys())  # 按位置标签排序
indices = range(len(locations))  # 创建对应的索引
values = [correlation_results[loc] for loc in locations]  # 获取排序后的相关系数

# 计算平均值
mean_value = np.mean(values)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.scatter(indices, values, edgecolor='blue', facecolors='none', s=100,label="Befor CNN-LSTM")  # 使用索引作为x轴

# 添加每个点到横轴的垂直线
for idx, value in zip(indices, values):
    plt.plot([idx, idx], [0, value], color='blue',  linestyle='--', linewidth=1)

# 添加平均值的水平虚线
plt.axhline(y=mean_value, color='r', linestyle='--')

plt.xticks(indices, [f"{loc}" for loc in locations], rotation=45)  # 设置x轴标签
plt.xlabel('Location Index')
plt.ylabel('Ambiguous Locations')
plt.legend()
plt.grid(True)
plt.show()