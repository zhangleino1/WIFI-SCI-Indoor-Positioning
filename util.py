
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#函数：中值滤波
def median_filter(data):
    # 按列应用中值滤波，假设kernel_size为3
    return medfilt(data, kernel_size=(1, 3))

# 函数：最小-最大规范化
def min_max_normalization(data):
        # 检查无穷大或NaN
    if np.isinf(data).any() or np.isnan(data).any():
        # 替换无穷大值为NaN
        data = np.where(np.isinf(data), np.nan, data)
        # 使用该列的最大值填充NaN
        max_per_column = np.nanmax(data, axis=0)
        data = np.where(np.isnan(data), max_per_column, data)
    # 按行应用最小-最大规范化
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data.T).T  # 转置是为了按行规范化