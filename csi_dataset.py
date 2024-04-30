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


import numpy as np
import pandas as pd
import os
import glob
import re
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import glob
import os
import re
from torch.utils.data import Dataset
from collections import OrderedDict

class CSIDataset(Dataset):
    def __init__(self, directory, time_step, stride=1, cache_size=270):
        self.directory = directory
        self.time_step = time_step
        self.stride = stride
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.data_files = {ant: sorted(glob.glob(os.path.join(self.directory, f'antenna_{ant}_*.csv'))) for ant in range(1, 4)}
        self.index_map = self._prepare_index_map()
        self.total_samples = sum(len(lst) for lst in self.index_map.values())

    def _prepare_index_map(self):
        index_map = {}
        for file_idx, file_path in enumerate(self.data_files[1]):
            df = self._load_csv(file_path)
            num_segments = (len(df[0]) - self.time_step + 1) // self.stride
            index_map[file_idx] = list(range(num_segments))
        return index_map

    def _load_csv(self, file_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测GPU
        key = file_path
        if key in self.cache:
            print(f"Cache hit for {file_path}")
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            print(f"Loading and processing {file_path}")
            df = pd.read_csv(file_path)
            amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
            phase_data = df.filter(regex='^phase_').values.astype(np.float32)
            
            # 应用中值滤波和最小-最大规范化
            amplitude_data = min_max_normalization(median_filter(amplitude_data))
            phase_data = min_max_normalization(median_filter(phase_data))
            
            # 转换为Tensor并移至GPU
            amplitude_tensor = torch.tensor(amplitude_data, dtype=torch.float32).to(device)
            phase_tensor = torch.tensor(phase_data, dtype=torch.float32).to(device)
            
            self.cache[key] = (amplitude_tensor, phase_tensor)  # 存储处理后的张量
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return self.cache[key]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative_count = 0
        for file_idx, segments in self.index_map.items():
            if cumulative_count + len(segments) > idx:
                segment_idx = idx - cumulative_count
                break
            cumulative_count += len(segments)

        all_channels_data = []
        for ant in range(1, 4):
            amplitude_tensor, phase_tensor = self._load_csv(self.data_files[ant][file_idx])
            start_idx = segment_idx * self.stride
            end_idx = start_idx + self.time_step
            all_channels_data.append(amplitude_tensor[start_idx:end_idx])
            all_channels_data.append(phase_tensor[start_idx:end_idx])
        
        sample_data = torch.stack(all_channels_data)
        match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(self.data_files[1][file_idx]))
        x, y = map(int, match.groups()) if match else (0, 0)
        return sample_data, torch.tensor([x, y], dtype=torch.float32, device=sample_data.device)



# class CSIDataset(Dataset):
#     def __init__(self, directory, time_step, stride=1):
#         self.directory = directory
#         self.time_step = time_step
#         self.stride = stride
#         self.data_cache = []  # Cache processed data
#         self._prepare_data()

#     def _prepare_data(self):
#         # Gather all files for each antenna into a list
#         antenna_files = {ant: sorted(glob.glob(os.path.join(self.directory, f'antenna_{ant}_*.csv'))) for ant in range(1, 4)}
        
#         # Check if all antennas have the same number of files
#         if not all(len(files) == len(antenna_files[1]) for files in antenna_files.values()):
#             raise ValueError("Mismatch in number of files across antennas")

#         # Process each file set
#         num_files = len(antenna_files[1])  # Assuming all antennas have the same number of files
#         for i in range(num_files):
#             file_set = [antenna_files[ant][i] for ant in range(1, 4)]
            
#             # Extract x, y from the first file's name (assuming all files in the set have the same location)
#             match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(file_set[0]))
#             if not match:
#                 continue  # Skip this set if the filename pattern doesn't match
            
#             x, y = map(int, match.groups())

#             # Prepare data for all antennas
#             all_channels_data = []
#             for file_path in file_set:
#                 df = pd.read_csv(file_path)
#                 amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)  # Convert to float
#                 phase_data = df.filter(regex='^phase_').values.astype(np.float32)  # Convert to float
#                               # 应用中值滤波和最小-最大规范化
#                 amplitude_data = min_max_normalization(median_filter(amplitude_data))
#                 phase_data = min_max_normalization(median_filter(phase_data))

#                 # Handle data segmentation with stride
#                 for start_idx in range(0, len(amplitude_data) - self.time_step + 1, self.stride):
#                     end_idx = start_idx + self.time_step
#                     segmented_amplitude = amplitude_data[start_idx:end_idx]
#                     segmented_phase = phase_data[start_idx:end_idx]

#                     all_channels_data.append(segmented_amplitude)
#                     all_channels_data.append(segmented_phase)

#             # Combine data into one sample per segment
#             for j in range(0, len(all_channels_data), 6):  # Process each group of 6 arrays (3 antennas x 2 data types)
#                 sample_data = np.stack(all_channels_data[j:j+6], axis=0)  # Ensure data stays as float32
#                 self.data_cache.append((sample_data, np.array([x, y], dtype=np.float32)))  # Ensure labels are also float

#     def __len__(self):
#         return len(self.data_cache)

#     def __getitem__(self, idx):
#         return self.data_cache[idx]


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
