import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os

from util import min_max_normalization, median_filter
import re
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import OrderedDict

class CSIDataset(Dataset):
    def __init__(self, directory, time_step, stride=1, cache_size=270):
        self.directory = directory
        self.time_step = time_step
        self.stride = stride
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.data_files = {ant: sorted(glob.glob(os.path.join(self.directory, f'antenna_{ant}_*.csv'))) for ant in range(1, 4)}
        
        # Extract all unique location classes (x,y coordinates)
        self.location_classes = set()
        for file_path in self.data_files[1]:
            match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(file_path))
            if match:
                x, y = map(int, match.groups())
                self.location_classes.add((x, y))
        
        # Create a mapping from (x,y) to class index
        self.location_to_class = {loc: idx for idx, loc in enumerate(sorted(self.location_classes))}
        self.class_to_location = {idx: loc for loc, idx in self.location_to_class.items()}
        self.num_classes = len(self.location_classes)
        
        print(f"Found {self.num_classes} unique location classes")
        
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
            df = pd.read_csv(file_path, na_values='#NAME?')  # Replace '#NAME?' with NaN
            amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
            phase_data = df.filter(regex='^phase_').values.astype(np.float32)
            
            # Apply median filter and min-max normalization
            amplitude_data = min_max_normalization(median_filter(amplitude_data))
            phase_data = min_max_normalization(median_filter(phase_data))
            
            # Convert to Tensor and move to GPU
            amplitude_tensor = torch.tensor(amplitude_data, dtype=torch.float32).to(device)
            phase_tensor = torch.tensor(phase_data, dtype=torch.float32).to(device)
            
            self.cache[key] = (amplitude_tensor, phase_tensor)  # Store processed tensors
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
        
        # Extract location from filename and convert to class index
        match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(self.data_files[1][file_idx]))
        if match:
            x, y = map(int, match.groups())
            location = (x, y)
            class_idx = self.location_to_class[location]
        else:
            class_idx = 0  # Default to class 0 if no match (shouldn't happen)
        
        return sample_data, torch.tensor(class_idx, dtype=torch.long, device=sample_data.device)

class CSIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, time_step, data_dir, stride):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_step = time_step
        self.data_dir = data_dir
        self.stride = stride

        # 加载全部数据索引
        self.dataset = CSIDataset(directory=self.data_dir, time_step=self.time_step, stride=self.stride)
        self.num_classes = self.dataset.num_classes  # Store number of classes for model initialization
        
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
