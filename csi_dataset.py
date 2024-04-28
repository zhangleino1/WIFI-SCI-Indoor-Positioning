import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from util import min_max_normalization, median_filter
import re
import torch
from sklearn.model_selection import train_test_split
import pandas as pd


class CSIDataset(Dataset):
    def __init__(self, directory, time_step, stride=1):
        self.directory = directory
        self.time_step = time_step
        self.stride = stride
        self.data_cache = []  # Cache processed data
        self._prepare_data()

    def _prepare_data(self):
        # Gather all files for each antenna into a list
        antenna_files = {ant: sorted(glob.glob(os.path.join(self.directory, f'antenna_{ant}_*.csv'))) for ant in range(1, 4)}
        
        # Check if all antennas have the same number of files
        if not all(len(files) == len(antenna_files[1]) for files in antenna_files.values()):
            raise ValueError("Mismatch in number of files across antennas")

        # Process each file set
        num_files = len(antenna_files[1])  # Assuming all antennas have the same number of files
        for i in range(num_files):
            file_set = [antenna_files[ant][i] for ant in range(1, 4)]
            
            # Extract x, y from the first file's name (assuming all files in the set have the same location)
            match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(file_set[0]))
            if not match:
                continue  # Skip this set if the filename pattern doesn't match
            
            x, y = map(int, match.groups())

            # Prepare data for all antennas
            all_channels_data = []
            for file_path in file_set:
                df = pd.read_csv(file_path)
                amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
                phase_data = df.filter(regex='^phase_').values.astype(np.float32)

                # Handle data segmentation with stride
                for start_idx in range(0, len(amplitude_data) - self.time_step + 1, self.stride):
                    end_idx = start_idx + self.time_step
                    segmented_amplitude = amplitude_data[start_idx:end_idx]
                    segmented_phase = phase_data[start_idx:end_idx]

                    all_channels_data.append(segmented_amplitude)
                    all_channels_data.append(segmented_phase)

            # Combine data into one sample per segment
            for j in range(0, len(all_channels_data), 6):  # Process each group of 6 arrays (3 antennas x 2 data types)
                sample_data = np.stack(all_channels_data[j:j+6])
                self.data_cache.append((sample_data, np.array([x, y], dtype=np.float32)))

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]



class CSIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, time_step, data_dir,stride):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_step = time_step
        self.data_dir = data_dir
        self.stride = stride

        # 加载全部数据索引
        self.dataset = CSIDataset(directory=self.data_dir, time_step=self.time_step, stride=self.stride)
        # 划分数据集
        self.train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=0.4, random_state=42)
        self.val_idx, self.test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_idx)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.test_idx)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=True)
