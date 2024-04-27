import os
import glob
import pandas as pd
import numpy as np
import re
from CSIKit.reader import get_reader
from CSIKit.util import csitools
# 用于解析.dat文件并保存为CSV文件，方便可视化看数据

# 指定.dat文件所在的目录
dat_directory = os.getcwd()+'/dataset/'

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


         # 为每个接收天线创建并保存一个CSV文件
        for rx_ant in range(3):  # 假设有三个接收天线
            filename = f"{x_label}_{y_label}_ant{rx_ant}.csv"
            csv_path = os.path.join(dat_directory, filename)

            # 如果文件已经存在，则跳过
            if os.path.exists(csv_path):
                continue

            # 获取幅度信息
            csi_amplitude = csitools.get_CSI(csi_data, metric="amplitude")[0][:, :, rx_ant, 0]
            csi_amplitude_squeezed = np.squeeze(csi_amplitude)

            # 获取相位信息
            csi_phase = csitools.get_CSI(csi_data, metric="phase")[0][:, :, rx_ant, 0]
            csi_phase_squeezed = np.squeeze(csi_phase)
            
            # 创建DataFrame
            amplitude_df = pd.DataFrame(csi_amplitude_squeezed, columns=[f"amplitude_{i}" for i in range(csi_amplitude_squeezed.shape[1])])
            phase_df = pd.DataFrame(csi_phase_squeezed, columns=[f"phase_{i}" for i in range(csi_phase_squeezed.shape[1])])

            # 合并数据帧
            combined_df = pd.concat([amplitude_df, phase_df], axis=1)
            
            # 加入时间戳
            timestamps = csi_data.timestamps if hasattr(csi_data, 'timestamps') else np.arange(len(csi_amplitude_squeezed))
            combined_df['timestamp'] = timestamps

            # 保存为CSV文件
            combined_df.to_csv(csv_path, index=False)
