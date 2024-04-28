import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from heatmappic import min_max_normalization, median_filter
import re
import torch
from sklearn.model_selection import train_test_split


class CSIDataset(Dataset):
    def __init__(self, directory, time_step, stride=1):
        self.files = glob.glob(os.path.join(directory, '*.dat'))
        self.time_step = time_step
        # 通过stride参数，你可以控制窗口的步进大小，这对于控制生成的样本数量和覆盖数据的方式非常有用
        self.stride = stride
        self.reader = get_reader()
        self.data_cache = []  # Cache processed data
        self._prepare_data()

    def _prepare_data(self):
        # Preprocess all data and cache it
        for file_path in self.files:
            # Read and process the data
            csi_data = self.reader.read_file(file_path, scaled=True)
            match = re.search(r'\((\d+),(\d+)\).dat', os.path.basename(file_path))
            x, y = map(int, match.groups()) if match else (0, 0)
            processed_data = []
            # 3个天线
            for rx_ant in range(3):
                csi_amplitude = csitools.get_CSI(csi_data, metric="amplitude")[0][:, :, rx_ant, 0]
                csi_phase = csitools.get_CSI(csi_data, metric="phase")[0][:, :, rx_ant, 0]

                # Normalize and filter
                amplitude_normalized = min_max_normalization(median_filter(csi_amplitude))
                phase_normalized = min_max_normalization(median_filter(csi_phase))

                # Extend the processed data list
                processed_data.extend([amplitude_normalized, phase_normalized])

            # Handle the time dimension
            data_length = processed_data[0].shape[1]
            num_samples = 1 + (data_length - self.time_step) // self.stride

            for start in range(0, num_samples * self.stride, self.stride):
                end = start + self.time_step
                if end <= data_length:
                    sample = [channel[:, start:end] for channel in processed_data]
                    self.data_cache.append((np.stack(sample, axis=0), np.array([x, y])))

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]





class CSIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, time_step, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_step = time_step
        self.data_dir = data_dir

        # 加载全部数据索引
        self.dataset = CSIDataset(directory=self.data_dir, time_step=self.time_step)
        self.train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=0.4, random_state=42)
        self.val_idx, self.test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    def setup(self, stage=None):
        # 根据索引划分数据集
        if stage == 'fit' or stage is None:
            self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_idx)
            self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_idx)

        if stage == 'test' or stage is None:
            self.test_dataset = torch.utils.data.Subset(self.dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
