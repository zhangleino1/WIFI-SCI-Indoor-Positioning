# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
import re

import torch
from sklearn.model_selection import train_test_split
import pandas as pd

from util import min_max_normalization, median_filter


class CSIDataset(Dataset):
    """
    Loads WiFi CSI data from CSV files and returns (sample, target) pairs
    for indoor positioning regression.

    Target format: float32 tensor [x, y] — the physical coordinates of the
    measurement location (in grid units).

    File naming: antenna_<antenna_id>_<x>_<y>.csv
    Each location must have antenna_1, antenna_2, and antenna_3 files.
    Each row in a CSV file represents one CSI measurement
    (30 amplitude + 30 phase columns).
    """

    def __init__(self, directory: str, time_step: int, stride: int = 1):
        self.directory = directory
        self.time_step = time_step
        self.stride    = stride
        self.data_cache: dict = {}

        # ---------- discover unique locations ----------
        all_files = glob.glob(os.path.join(directory, 'antenna_*.csv'))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in {directory} matching 'antenna_*.csv'")

        location_classes: set = set()
        for fp in all_files:
            m = re.search(r'antenna_\d+_(\d+)_(\d+)\.csv', os.path.basename(fp))
            if m:
                location_classes.add((int(m.group(1)), int(m.group(2))))

        if not location_classes:
            raise ValueError(f"Could not parse any locations from file names in {directory}")

        # Stable sorted mapping (kept for debugging / visualisation tools)
        self.location_to_class = {loc: idx for idx, loc in enumerate(sorted(location_classes))}
        self.class_to_location = {idx: loc for loc, idx in self.location_to_class.items()}
        self.num_classes = len(location_classes)
        print(f"Found {self.num_classes} unique locations: {sorted(location_classes)}")

        self._load_all_data()
        self._prepare_index_map()

    # ------------------------------------------------------------------

    def _load_all_data(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for location in self.location_to_class:
            x, y = location
            file_paths = [
                os.path.join(self.directory, f'antenna_1_{x}_{y}.csv'),
                os.path.join(self.directory, f'antenna_2_{x}_{y}.csv'),
                os.path.join(self.directory, f'antenna_3_{x}_{y}.csv'),
            ]
            if not all(os.path.exists(fp) for fp in file_paths):
                print(f"Warning: Missing antenna files for location {location}. Skipping.")
                continue

            location_data = []
            for fp in file_paths:
                print(f"Loading {fp}")
                df  = pd.read_csv(fp, na_values='#NAME?')
                amp = df.filter(regex='^amplitude_').values.astype(np.float32)
                pha = df.filter(regex='^phase_').values.astype(np.float32)

                if amp.shape[0] == 0 or pha.shape[0] == 0:
                    raise ValueError(f"Empty data in {fp}")

                amp = min_max_normalization(median_filter(amp))
                pha = min_max_normalization(median_filter(pha))

                location_data.append((
                    torch.tensor(amp, dtype=torch.float32).to(device),
                    torch.tensor(pha, dtype=torch.float32).to(device),
                ))
            self.data_cache[location] = location_data

    def _prepare_index_map(self):
        self.sample_info_list: list = []
        for location, location_data in self.data_cache.items():
            num_rows = location_data[0][0].shape[0]
            if num_rows < self.time_step:
                print(f"Warning: location {location} has only {num_rows} rows "
                      f"(< time_step={self.time_step}). Skipping.")
                continue
            num_segments = (num_rows - self.time_step + 1) // self.stride
            if num_segments <= 0:
                continue
            for seg in range(num_segments):
                self.sample_info_list.append((location, seg))

        self.total_samples = len(self.sample_info_list)
        if self.total_samples == 0:
            raise ValueError("No valid samples. Check data directory and time_step/stride.")
        print(f"Prepared {self.total_samples} total samples.")

    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        location, seg_idx = self.sample_info_list[idx]
        location_data     = self.data_cache[location]

        start = seg_idx * self.stride
        end   = start + self.time_step

        channels = []
        for amp_t, pha_t in location_data:           # 3 antennas
            channels.append(amp_t[start:end, :])
            channels.append(pha_t[start:end, :])

        sample_data = torch.stack(channels)           # (6, time_step, num_subcarriers)

        if idx == 0:
            print(f"CSIDataset sample[0]: shape={sample_data.shape}, location={location}, "
                  f"min={sample_data.min():.4f}, max={sample_data.max():.4f}")

        x_coord, y_coord = location
        target = torch.tensor([float(x_coord), float(y_coord)],
                               dtype=torch.float32, device=sample_data.device)
        return sample_data, target


# ---------------------------------------------------------------------------

class CSIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that loads CSI data and splits it into
    train (60%) / val (20%) / test (20%) subsets.
    """

    def __init__(self, batch_size: int, num_workers: int,
                 time_step: int, data_dir: str, stride: int):
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.time_step   = time_step
        self.data_dir    = data_dir
        self.stride      = stride

        self.dataset     = CSIDataset(directory=data_dir, time_step=time_step, stride=stride)
        self.num_classes = self.dataset.num_classes   # kept for visualisation tools

        # Print sample counts per location
        print("\n--- Location Sample Counts ---")
        loc_counts: dict = {}
        for location, _ in self.dataset.sample_info_list:
            loc_counts[location] = loc_counts.get(location, 0) + 1
        for loc in sorted(loc_counts):
            print(f"  Location {loc}: {loc_counts[loc]} samples")
        print(f"  Total: {self.dataset.total_samples}\n")

        # Train / val / test split
        indices = list(range(len(self.dataset)))
        train_idx, tmp_idx = train_test_split(indices, test_size=0.4, random_state=42)
        val_idx,   test_idx = train_test_split(tmp_idx,  test_size=0.5, random_state=42)

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        self.val_dataset   = torch.utils.data.Subset(self.dataset, val_idx)
        self.test_dataset  = torch.utils.data.Subset(self.dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)
