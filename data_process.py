import os
import glob
import pandas as pd
import numpy as np
import re
from CSIKit.reader import get_reader
from CSIKit.util import csitools

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
        filename = f"{x_label}_{y_label}.csv"  # 创建以坐标命名的文件名

        # 读取CSI数据
        my_reader = get_reader(dat_file)
        csi_data = my_reader.read_file(dat_file, scaled=True)
        
        # 获取幅度信息
        csi_amplitude, _, _ = csitools.get_CSI(csi_data, metric="amplitude")
        csi_amplitude_first = csi_amplitude[:, :, 0, 0]
        csi_amplitude_squeezed = np.squeeze(csi_amplitude_first)

        # 获取相位信息
        csi_phase, _, _ = csitools.get_CSI(csi_data, metric="phase")
        csi_phase_first = csi_phase[:, :, 0, 0]
        csi_phase_squeezed = np.squeeze(csi_phase_first)
        
        # 创建DataFrame
        amplitude_df = pd.DataFrame(csi_amplitude_squeezed, columns=[f"amplitude_{i}" for i in range(csi_amplitude_squeezed.shape[1])])
        phase_df = pd.DataFrame(csi_phase_squeezed, columns=[f"phase_{i}" for i in range(csi_phase_squeezed.shape[1])])

        # 合并数据帧
        combined_df = pd.concat([amplitude_df, phase_df], axis=1)
        
        # 加入时间戳
        timestamps = csi_data.timestamps if hasattr(csi_data, 'timestamps') else np.arange(len(csi_amplitude_squeezed))
        combined_df['timestamp'] = timestamps

        # 保存为CSV文件
        csv_path = os.path.join(dat_directory, filename)
        combined_df.to_csv(csv_path, index=False)
