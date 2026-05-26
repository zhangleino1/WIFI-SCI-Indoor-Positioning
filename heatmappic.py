# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import argparse
import os
import glob

import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util import min_max_normalization
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False 

# 函数：绘制和保存热力图
def save_heatmap(data, file_name, title):
    # 检查文件是否已存在
    if os.path.exists(file_name):
        print(f"{file_name} 已存在，跳过生成。")
        return
    plt.figure(figsize=(10, 8))

    data_array = np.asarray(data)
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
    plt.title(title + "热力图")
    plt.savefig(file_name)
    plt.close()
    print(f"热力图已保存为{file_name}")


def create_heatmaps(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(data_dir, 'CSI_RX_*.mat')))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir} matching 'CSI_RX_*.mat'")

    for mat_path in mat_files:
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        with h5py.File(mat_path, 'r') as mat_file:
            amp_all = mat_file['csiAmplitudeFiltered'][()].astype(np.float32)
            pha_all = mat_file['csiPhaseCalibrated'][()].astype(np.float32)

        if amp_all.shape != pha_all.shape:
            raise ValueError(
                f"Amplitude/phase shape mismatch in {mat_path}: "
                f"{amp_all.shape} vs {pha_all.shape}")

        for ant_idx in range(amp_all.shape[0]):
            amp = amp_all[ant_idx].T
            pha = pha_all[ant_idx].T
            amp_norm = min_max_normalization(amp)
            pha_norm = min_max_normalization(pha)

            save_heatmap(
                amp_norm,
                os.path.join(output_dir, f"{base_name}_ant{ant_idx}_amplitude.png"),
                f"Ant{ant_idx} 幅度",
            )
            save_heatmap(
                pha_norm,
                os.path.join(output_dir, f"{base_name}_ant{ant_idx}_phase.png"),
                f"Ant{ant_idx} 相位",
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate heatmaps for .mat CSI data')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'dataset'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'results', 'heatmaps'))
    args = parser.parse_args()
    create_heatmaps(args.data_dir, args.output_dir)
