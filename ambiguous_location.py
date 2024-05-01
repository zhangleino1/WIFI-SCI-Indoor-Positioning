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
                    # 将三个天线的数据沿着时间步（第一维）连接起来
                    location_data = np.concatenate(all_data, axis=0)
                    data_by_location[location_key] = location_data
        return data_by_location

    def calculate_physical_distance(self,loc1, loc2):
        x1, y1 = map(int, loc1)
        x2, y2 = map(int, loc2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
   
    def calculate_inter_location_correlation(self, data_by_location):
        correlation_results = {}
        locations = list(data_by_location.keys())  # 获取所有位置的列表
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):  # 避免重复计算和自我比较
                data_i = data_by_location[locations[i]]
                data_j = data_by_location[locations[j]]
                coeffs = []
                # 假设data_i和data_j是同样形状的数组
                for channel in range(min(data_i.shape[0], data_j.shape[0])):  # 遍历频道
                    xi = data_i[channel, :, :]  # 选择特定频道的所有时间步和特征
                    xj = data_j[channel, :, :]
                    coeff, _ = pearsonr(xi.flatten(), xj.flatten())
                    if not np.isnan(coeff) and not np.isinf(coeff):
                        coeffs.append(coeff)
                    else:
                        print(f"Found NaN or Inf values in correlation between {locations[i]} and {locations[j]}")
                correlation_results[(locations[i], locations[j])] = np.mean(coeffs)
        return correlation_results
    
    def identify_ambiguous_locations(self, data_by_location, correlation_results, grid_size=0.5, avg_corrs=0.8):
        ambiguous_points = {}
        locations = list(data_by_location.keys())

        for i, loc_i in enumerate(locations):
            ambiguous_for_i = []
            for j, loc_j in enumerate(locations):
                if i != j:  # 确保不与自身比较
                    distance = self.calculate_physical_distance(loc_i, loc_j)
                    # 从correlation_results中获取正确的相关性值
                    correlation = correlation_results.get((loc_i, loc_j), correlation_results.get((loc_j, loc_i), 0))
                    # 如果距离大于网格大小并且相关系数高于平均值，则视为模糊点
                    if distance > grid_size and correlation > avg_corrs:
                        ambiguous_for_i.append(loc_j)
            ambiguous_points[loc_i] = ambiguous_for_i

        return ambiguous_points



# 使用
data_directory = data_dir=os.getcwd()+"/dataset_test"
handler = CSIDataHandler(data_directory)
data_by_location = handler.load_data_by_location()
correlation_results = handler.calculate_inter_location_correlation(data_by_location)
# 使用数据和相关系数来识别模糊点
ambiguous_locations = handler.identify_ambiguous_locations(data_by_location, correlation_results)


# 计算每个位置的模糊点数量
ambiguous_counts = {loc: len(amb_locs) for loc, amb_locs in ambiguous_locations.items()}

# 准备数据
locations = sorted(ambiguous_counts.keys())  # 按位置标签排序
indices = range(len(locations))  # 创建对应的索引
values = [ambiguous_counts[loc] for loc in locations]  # 获取排序后的模糊点数量

# 计算平均值
mean_value = np.mean(values)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.scatter(indices, values, edgecolor='blue', facecolors='none', s=100, label="Before CNN-LSTM")  # 使用索引作为x轴

# 添加每个点到横轴的垂直线
for idx, value in zip(indices, values):
    plt.plot([idx, idx], [0, value], color='blue', linestyle='--', linewidth=1)

# 添加平均值的水平虚线
plt.axhline(y=mean_value, color='r', linestyle='--', label="Mean CNN-LSTM")

# plt.xticks(indices, [str(loc) for loc in locations], rotation=45)  # 使用位置标签作为x轴标签
plt.xticks(indices, [index for index in indices], rotation=45)
plt.xlabel('Location Index')
plt.ylabel('Ambiguous Locations')
plt.legend()
plt.grid(True)
plt.show()