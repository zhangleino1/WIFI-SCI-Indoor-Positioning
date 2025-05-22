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
        
        # Extract all unique location classes (x,y coordinates)
        # Scan directory for all files to find unique locations
        self.location_classes = set()
        all_files = glob.glob(os.path.join(self.directory, 'antenna_*.csv'))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in directory {self.directory} with pattern 'antenna_*.csv'")
            
        for file_path in all_files:
            match = re.search(r'antenna_\d+_(\d+)_(\d+).csv', os.path.basename(file_path))
            if match:
                x, y = map(int, match.groups())
                self.location_classes.add((x, y))
        
        if not self.location_classes:
            raise ValueError(f"No locations found from file names in {self.directory}")

        # Create a mapping from (x,y) to class index
        self.location_to_class = {loc: idx for idx, loc in enumerate(sorted(self.location_classes))}
        self.class_to_location = {idx: loc for loc, idx in self.location_to_class.items()}
        self.num_classes = len(self.location_classes)
        
        print(f"Found {self.num_classes} unique location classes: {sorted(self.location_classes)}")
        
        self._prepare_index_map() # This will populate self.sample_info_list and self.total_samples

    def _prepare_index_map(self):
        self.sample_info_list = []
        for location in self.location_to_class.keys():
            x, y = location
            file_path_ant1 = os.path.join(self.directory, f'antenna_1_{x}_{y}.csv')
            file_path_ant2 = os.path.join(self.directory, f'antenna_2_{x}_{y}.csv')
            file_path_ant3 = os.path.join(self.directory, f'antenna_3_{x}_{y}.csv')

            if os.path.exists(file_path_ant1) and \
               os.path.exists(file_path_ant2) and \
               os.path.exists(file_path_ant3):
                
                try:
                    # Efficiently get row count from the first antenna's CSV
                    # Assuming all antenna files for the same location have the same number of rows
                    num_rows = pd.read_csv(file_path_ant1, usecols=[0]).shape[0]
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty CSV file encountered: {file_path_ant1}. Skipping location {location}.")
                    continue
                except Exception as e:
                    print(f"Warning: Error reading {file_path_ant1}: {e}. Skipping location {location}.")
                    continue
                
                if num_rows < self.time_step:
                    print(f"Warning: Not enough data in {file_path_ant1} for time_step {self.time_step}. Has {num_rows} rows. Skipping location {location}.")
                    continue

                num_segments = (num_rows - self.time_step + 1) // self.stride
                if num_segments <= 0:
                    print(f"Warning: No segments generated for {file_path_ant1} with {num_rows} rows, time_step {self.time_step}, stride {self.stride}. Skipping location {location}.")
                    continue
                    
                for segment_idx in range(num_segments):
                    self.sample_info_list.append(
                        (file_path_ant1, file_path_ant2, file_path_ant3, segment_idx, location)
                    )
            else:
                print(f"Warning: Missing one or more antenna files for location {location}. Skipping this location.")
        
        self.total_samples = len(self.sample_info_list)
        if self.total_samples == 0:
            raise ValueError("No valid samples found. Check data directory, file naming, and time_step/stride parameters.")
        print(f"Prepared {self.total_samples} total samples.")

    def _load_csv(self, file_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        key = file_path
        if key in self.cache:
            # print(f"Cache hit for {file_path}") # Reduce verbosity
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            # print(f"Loading and processing {file_path}") # Reduce verbosity
            df = pd.read_csv(file_path, na_values='#NAME?')
            amplitude_data = df.filter(regex='^amplitude_').values.astype(np.float32)
            phase_data = df.filter(regex='^phase_').values.astype(np.float32)
            
            if amplitude_data.shape[0] == 0 or phase_data.shape[0] == 0:
                raise ValueError(f"Empty data after filtering in file: {file_path}")

            amplitude_data = min_max_normalization(median_filter(amplitude_data))
            phase_data = min_max_normalization(median_filter(phase_data))
            
            amplitude_tensor = torch.tensor(amplitude_data, dtype=torch.float32).to(device)
            phase_tensor = torch.tensor(phase_data, dtype=torch.float32).to(device)
            
            self.cache[key] = (amplitude_tensor, phase_tensor)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return self.cache[key]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_path_ant1, file_path_ant2, file_path_ant3, segment_idx, location = self.sample_info_list[idx]

        amplitude_tensor_ant1, phase_tensor_ant1 = self._load_csv(file_path_ant1)
        amplitude_tensor_ant2, phase_tensor_ant2 = self._load_csv(file_path_ant2)
        amplitude_tensor_ant3, phase_tensor_ant3 = self._load_csv(file_path_ant3)

        start_idx = segment_idx * self.stride
        end_idx = start_idx + self.time_step

        all_channels_data = [
            amplitude_tensor_ant1[start_idx:end_idx, :],
            phase_tensor_ant1[start_idx:end_idx, :],
            amplitude_tensor_ant2[start_idx:end_idx, :],
            phase_tensor_ant2[start_idx:end_idx, :],
            amplitude_tensor_ant3[start_idx:end_idx, :],
            phase_tensor_ant3[start_idx:end_idx, :]
        ]
        
        # Ensure all segments have the expected time_step length
        for i, segment in enumerate(all_channels_data):
            if segment.shape[0] != self.time_step:
                raise ValueError(f"Segment {i} for sample {idx} (loc {location}, seg_idx {segment_idx}) has incorrect time_step {segment.shape[0]}, expected {self.time_step}. File {self.sample_info_list[idx][i//2]}. Start {start_idx}, End {end_idx}")

        sample_data = torch.stack(all_channels_data) # Shape: (6, time_step, num_subcarriers)
        
        class_idx = self.location_to_class[location]
        
        if idx == 0:
            print(f"CSIDataset: sample_data shape: {sample_data.shape}, class_idx: {class_idx}")
            # Example: CSIDataset: sample_data shape: torch.Size([6, 15, 30]), class_idx: 0
            # This means 6 channels, 15 time steps (height), 30 subcarriers (width)

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

        # --- Class Distribution Analysis ---
        if hasattr(self.dataset, 'sample_info_list') and self.dataset.sample_info_list:
            print("\n--- Class Distribution Analysis (Full Dataset) ---")
            class_counts = {i: 0 for i in range(self.num_classes)}
            
            for sample_info in self.dataset.sample_info_list:
                # sample_info is (file_path_ant1, file_path_ant2, file_path_ant3, segment_idx, location)
                location = sample_info[4] 
                class_idx = self.dataset.location_to_class[location]
                class_counts[class_idx] += 1
            
            total_samples_from_distribution = 0
            for i in range(self.num_classes):
                location_coord = self.dataset.class_to_location[i]
                count = class_counts[i]
                print(f"Class {i} (Location: {location_coord}): {count} samples")
                total_samples_from_distribution += count
            
            print(f"Total samples in distribution: {total_samples_from_distribution}")
            # Verification:
            if total_samples_from_distribution != self.dataset.total_samples:
                print(f"Warning: Mismatch in total samples! Distribution sum: {total_samples_from_distribution}, CSIDataset.total_samples: {self.dataset.total_samples}")
            print("--- End of Class Distribution Analysis ---\n")
        else:
            print("Warning: self.dataset.sample_info_list not available or empty. Skipping class distribution analysis.")
        # --- End of Class Distribution Analysis ---

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
