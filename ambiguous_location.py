import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import torch
import torch.nn as nn
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


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNN_LSTM_Net(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.test_losses = []  # 初始化空列表以收集测试损失
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # LSTM层
        self.lstm = nn.LSTM(input_size=18*30, hidden_size=100, batch_first=True)

        # 输出层
        self.fc = nn.Linear(100, 2)

        # 用于存储LSTM特征的属性
        self.lstm_features = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 重塑输出以匹配LSTM输入 (batch_size, seq_len, features)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        self.lstm_features = lstm_out[:, -1, :]  # 存储最后一个时间步的特征
        
        # 经过一个全连接层输出最终的二维坐标
        x = self.fc(self.lstm_features)
        return self.lstm_features


        

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Net().to(device)
# 注意修改路径
model_path = os.getcwd() + '/logs/cnn_lstm/version_9/checkpoints/last.ckpt'
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()  # 设置为评估模式



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
    
   
    def calculate_inter_location_correlation(self, data_by_location, type='raw'):
        correlation_results = {}
        locations = list(data_by_location.keys())  # 获取所有位置的列表
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):  # 避免重复计算和自我比较
                data_i = data_by_location[locations[i]]
                data_j = data_by_location[locations[j]]
                coeffs = []
                num_channels = min(data_i.shape[0], data_j.shape[0])
                step_size = 6

                for start in range(0, num_channels, step_size):
                    end = min(start + step_size, num_channels)  # 确保不越界

                    # 截取整个区间的数据
                    if type == 'raw':
                        xi = data_i[start:end, :, :].reshape(-1)  # 将选定区间的所有数据合并为一维数组
                        xj = data_j[start:end, :, :].reshape(-1)
                    elif type == 'cnn_lstm':
                        # 对每个区间的数据增加一个批量维度，以符合模型输入的需求
                        xi_tensor = torch.tensor(data_i[start:end, :, :], dtype=torch.float32).unsqueeze(0).to(device)
                        xj_tensor = torch.tensor(data_j[start:end, :, :], dtype=torch.float32).unsqueeze(0).to(device)
                        xi = model(xi_tensor).detach().cpu().numpy().flatten()  # 从模型获取数据并展平
                        xj = model(xj_tensor).detach().cpu().numpy().flatten()
                    coeff, _ = pearsonr(xi, xj)
                    if not np.isnan(coeff) and not np.isinf(coeff):
                        coeffs.append(coeff)
                    else:
                        print(f"Found NaN or Inf values in correlation between {locations[i]} and {locations[j]}")

                # 只在有有效系数时计算平均值
                if coeffs:
                    correlation_results[(locations[i], locations[j])] = np.mean(coeffs)
                else:
                    correlation_results[(locations[i], locations[j])] = 0

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

# 原始数据的相关性
correlation_results = handler.calculate_inter_location_correlation(data_by_location,type="raw")
# 使用数据和相关系数来识别模糊点
ambiguous_locations = handler.identify_ambiguous_locations(data_by_location, correlation_results)

# lstm-cnn数据的相关性
correlation_results_cnn_lstm = handler.calculate_inter_location_correlation(data_by_location,type="cnn_lstm")
# 使用数据和相关系数来识别模糊点
ambiguous_locations_cnn_lstm = handler.identify_ambiguous_locations(data_by_location, correlation_results_cnn_lstm)


# 计算每个位置的模糊点数量
ambiguous_counts = {loc: len(amb_locs) for loc, amb_locs in ambiguous_locations.items()}
# 计算LSTM-CNN数据的模糊点数量
ambiguous_counts_cnn_lstm = {loc: len(amb_locs) for loc, amb_locs in ambiguous_locations_cnn_lstm.items()}

# 准备数据
locations = sorted(set(ambiguous_counts.keys()).union(set(ambiguous_counts_cnn_lstm.keys())))  # 合并所有位置并排序
indices = range(len(locations))  # 创建对应的索引
values_raw = [ambiguous_counts.get(loc, 0) for loc in locations]  # 获取原始数据排序后的模糊点数量
values_cnn_lstm = [ambiguous_counts_cnn_lstm.get(loc, 0) for loc in locations]  # 获取LSTM-CNN数据排序后的模糊点数量

# 计算平均值
mean_value_raw = np.mean(values_raw)
mean_value_cnn_lstm = np.mean(values_cnn_lstm)

# 可视化结果
plt.figure(figsize=(15, 8))
plt.scatter(indices, values_raw, edgecolor='blue', facecolors='none', s=100, label="Before CNN-LSTM")  # 原始数据
plt.scatter(indices, values_cnn_lstm, edgecolor='green', facecolors='none', s=100, label="After CNN-LSTM")  # LSTM-CNN处理后的数据

# 添加每个点到横轴的垂直线
for idx, (value_raw, value_cnn_lstm) in enumerate(zip(values_raw, values_cnn_lstm)):
    plt.plot([idx, idx], [0, value_raw], color='blue', linestyle='--', linewidth=1)
    plt.plot([idx, idx], [0, value_cnn_lstm], color='green', linestyle='--', linewidth=1)

# 添加平均值的水平虚线
plt.axhline(y=mean_value_raw, color='blue', linestyle='--', label="Mean Before CNN-LSTM")
plt.axhline(y=mean_value_cnn_lstm, color='green', linestyle='--', label="Mean After CNN-LSTM")

plt.xticks(indices, [index for index in indices], rotation=45)  # 使用位置标签作为x轴标签
plt.xlabel('Location Index')
plt.ylabel('Number of Ambiguous Locations')
plt.ylabel('Ambiguous Locations')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.getcwd() + '/ambiguous_locations.png')  # 保存图像