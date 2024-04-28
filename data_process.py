import os
import glob
import pandas as pd
import numpy as np
import re
from CSIKit.reader import get_reader
from CSIKit.util import csitools

# 指定.dat文件所在的目录
dat_directory = os.getcwd() + '/dataset/'

# 使用glob找到所有的.dat文件
dat_files = glob.glob(os.path.join(dat_directory, '*.dat'))

# 遍历文件列表
for dat_file in dat_files:
    # 从文件名中提取坐标
    match = re.search(r'\((\d+),(\d+)\).dat', os.path.basename(dat_file))
    if match:
        x_label, y_label = match.groups()

        # 读取CSI数据
        my_reader = get_reader(dat_file)
        csi_data = my_reader.read_file(dat_file, scaled=True)
        
        for rx_ant in range(3):  # 假设有三个接收天线
            # 创建以天线和坐标命名的CSV文件名
            filename = f"antenna_{rx_ant+1}_{x_label}_{y_label}.csv"
            csv_path = os.path.join(dat_directory, filename)

            # 如果文件已经存在，则跳过
            if os.path.exists(csv_path):
                print(f"{filename} already exists. Skipping.")
                continue

            # 获取幅度和相位信息
            csi_amplitude = csitools.get_CSI(csi_data, metric="amplitude")[0][:, :, rx_ant, 0]
            csi_phase = csitools.get_CSI(csi_data, metric="phase")[0][:, :, rx_ant, 0]
            
            # 创建DataFrame
            amplitude_df = pd.DataFrame(csi_amplitude, columns=[f"amplitude_{i}" for i in range(csi_amplitude.shape[1])])
            phase_df = pd.DataFrame(csi_phase, columns=[f"phase_{i}" for i in range(csi_phase.shape[1])])
            
            # 合并DataFrame
            combined_df = pd.concat([amplitude_df, phase_df], axis=1)
            
            # 加入时间戳
            timestamps = csi_data.timestamps if hasattr(csi_data, 'timestamps') else np.arange(len(csi_amplitude))
            combined_df['timestamp'] = timestamps
            
            # 将DataFrame保存为CSV文件
            combined_df.to_csv(csv_path, index=False)
            print(f"Saved data to {filename}")
