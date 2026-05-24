# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import h5py
import os
import re
from typing import Optional

import torch
from sklearn.model_selection import train_test_split

from util import min_max_normalization


class CSIDataset(Dataset):
    """
    Loads WiFi CSI data from MATLAB v7.3 .mat files and returns (sample, target) pairs
    for indoor positioning regression.

    Target format: float32 tensor [x, y] — the physical coordinates of the
    measurement location (in grid units).

    File naming: CSI_RX_x_p1d000_y_p3d000_z_p0d000_<date>_<time>.mat
    Each file contains all antennas for one measurement location.
    """

    def __init__(self, directory: str, time_step: int, stride: int = 1):
        self.directory = directory
        self.time_step = time_step
        self.stride    = stride
        self.data_cache: dict = {}
        self.num_subcarriers: Optional[int] = None
        self.num_antennas: Optional[int] = None
        self.in_channels: Optional[int] = None

        all_files = sorted(glob.glob(os.path.join(directory, 'CSI_RX_*.mat')))
        if not all_files:
            raise FileNotFoundError(
                f"No .mat files found in {directory} matching 'CSI_RX_*.mat'")

        # Build location → file mapping from filenames only (no I/O).
        # Coordinate validation against the .mat internal coords is deferred
        # to _load_all_data so each file is opened exactly once.
        location_files: dict = {}
        for fp in all_files:
            file_location = self._parse_location_from_filename(fp)
            if file_location in location_files:
                raise ValueError(
                    f"Duplicate location {file_location} in {fp} "
                    f"and {location_files[file_location]}")
            location_files[file_location] = fp

        if not location_files:
            raise ValueError(f"Could not discover any locations in {directory}")

        self.location_files = location_files
        self.locations = sorted(location_files)
        self.num_locations = len(self.locations)
        print(f"Found {self.num_locations} unique locations: {self.locations}")

        self._load_all_data()
        self._prepare_index_map()

    # ------------------------------------------------------------------

    @staticmethod
    def _decode_coord_token(token: str) -> float:
        m = re.fullmatch(r'([pn])(\d+)d(\d+)', token)
        if not m:
            raise ValueError(f"Invalid coordinate token: {token}")

        sign = 1.0 if m.group(1) == 'p' else -1.0
        value = float(f"{m.group(2)}.{m.group(3)}")
        return sign * value

    @classmethod
    def _parse_location_from_filename(cls, file_path: str):
        name = os.path.basename(file_path)
        m = re.search(r'CSI_RX_x_([pn]\d+d\d+)_y_([pn]\d+d\d+)_z_', name)
        if not m:
            raise ValueError(f"Could not parse location from file name: {name}")
        return (
            cls._decode_coord_token(m.group(1)),
            cls._decode_coord_token(m.group(2)),
        )

    @staticmethod
    def _read_scalar(mat_file, key: str) -> float:
        if key not in mat_file:
            raise KeyError(f"Missing required key '{key}'")
        value = mat_file[key][()]
        return float(np.asarray(value).squeeze())

    def _load_all_data(self):
        for location in self.locations:
            fp = self.location_files[location]
            print(f"Loading {fp}")
            location_data = []

            with h5py.File(fp, 'r') as mat_file:
                # Validate filename coordinates against .mat internal coords
                mat_x = self._read_scalar(mat_file, 'coord/x')
                mat_y = self._read_scalar(mat_file, 'coord/y')
                if not np.allclose(location, (mat_x, mat_y), atol=1e-6):
                    raise ValueError(
                        f"Location mismatch in {fp}: "
                        f"filename={location}, mat=({mat_x}, {mat_y})")

                amp_all = mat_file['csiAmplitudeFiltered'][()].astype(np.float32)
                pha_all = mat_file['csiPhaseCalibrated'][()].astype(np.float32)

            if amp_all.shape != pha_all.shape:
                raise ValueError(
                    f"Amplitude/phase shape mismatch in {fp}: "
                    f"{amp_all.shape} vs {pha_all.shape}")
            if amp_all.ndim != 3:
                raise ValueError(
                    f"Expected CSI arrays with shape "
                    f"(n_antennas, n_subcarriers, n_packets), got {amp_all.shape} in {fp}")

            num_antennas, num_subcarriers, num_packets = amp_all.shape
            if self.num_antennas is None:
                self.num_antennas = num_antennas
                self.in_channels = num_antennas * 2
            elif num_antennas != self.num_antennas:
                raise ValueError(
                    f"Inconsistent num_antennas in {fp}: expected "
                    f"{self.num_antennas}, got {num_antennas}")

            if self.num_subcarriers is None:
                self.num_subcarriers = num_subcarriers
            elif num_subcarriers != self.num_subcarriers:
                raise ValueError(
                    f"Inconsistent num_subcarriers in {fp}: expected "
                    f"{self.num_subcarriers}, got {num_subcarriers}")

            if num_packets == 0:
                raise ValueError(f"Empty packet dimension in {fp}")

            for ant_idx in range(num_antennas):
                amp = amp_all[ant_idx].T
                pha = pha_all[ant_idx].T
                amp = min_max_normalization(amp)
                pha = min_max_normalization(pha)
                location_data.append((
                    torch.tensor(amp, dtype=torch.float32),
                    torch.tensor(pha, dtype=torch.float32),
                ))
            self.data_cache[location] = location_data

        if self.num_subcarriers is None:
            raise ValueError("Could not infer num_subcarriers from dataset.")
        if self.in_channels is None:
            raise ValueError("Could not infer in_channels from dataset.")

    def _prepare_index_map(self):
        self.sample_info_list: list = []
        for location, location_data in self.data_cache.items():
            num_rows = location_data[0][0].shape[0]
            if num_rows < self.time_step:
                print(f"Warning: location {location} has only {num_rows} rows "
                      f"(< time_step={self.time_step}). Skipping.")
                continue
            for start in range(0, num_rows - self.time_step + 1, self.stride):
                self.sample_info_list.append((location, start))

        self.total_samples = len(self.sample_info_list)
        if self.total_samples == 0:
            raise ValueError("No valid samples. Check data directory and time_step/stride.")
        print(f"Prepared {self.total_samples} total samples.")

    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        location, start = self.sample_info_list[idx]
        location_data     = self.data_cache[location]
        end   = start + self.time_step

        channels = []
        for amp_t, pha_t in location_data:
            channels.append(amp_t[start:end, :])
            channels.append(pha_t[start:end, :])

        sample_data = torch.stack(channels)           # (channels, time_step, num_subcarriers)

        x_coord, y_coord = location
        target = torch.tensor([float(x_coord), float(y_coord)], dtype=torch.float32)
        return sample_data, target


# ---------------------------------------------------------------------------

class CSIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that loads CSI data and splits it into
    train (60%) / val (20%) / test (20%) subsets.
    """

    def __init__(self, batch_size: int, num_workers: int,
                 time_step: int, data_dir: str, stride: int,
                 split_mode: str = 'by_location', split_seed: int = 42):
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.time_step   = time_step
        self.data_dir    = data_dir
        self.stride      = stride
        self.split_mode  = split_mode
        self.split_seed  = split_seed

        self.dataset     = CSIDataset(directory=data_dir, time_step=time_step, stride=stride)
        self.num_locations = self.dataset.num_locations
        self.num_subcarriers = self.dataset.num_subcarriers
        self.in_channels = self.dataset.in_channels

        # Print sample counts per location
        print("\n--- Location Sample Counts ---")
        loc_counts: dict = {}
        for location, _ in self.dataset.sample_info_list:
            loc_counts[location] = loc_counts.get(location, 0) + 1
        for loc in sorted(loc_counts):
            print(f"  Location {loc}: {loc_counts[loc]} samples")
        print(f"  Total: {self.dataset.total_samples}\n")

        if self.split_mode == 'random':
            indices = list(range(len(self.dataset)))
            train_idx, tmp_idx = train_test_split(
                indices, test_size=0.4, random_state=self.split_seed)
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=0.5, random_state=self.split_seed)
        elif self.split_mode == 'by_location':
            valid_locations = sorted({location for location, _ in self.dataset.sample_info_list})
            if len(valid_locations) < 3:
                raise ValueError("Need at least 3 valid locations for by_location split.")

            train_locs, tmp_locs = train_test_split(
                valid_locations, test_size=0.4, random_state=self.split_seed)
            val_locs, test_locs = train_test_split(
                tmp_locs, test_size=0.5, random_state=self.split_seed)

            train_locs = set(train_locs)
            val_locs = set(val_locs)
            test_locs = set(test_locs)

            train_idx = [
                idx for idx, (location, _) in enumerate(self.dataset.sample_info_list)
                if location in train_locs
            ]
            val_idx = [
                idx for idx, (location, _) in enumerate(self.dataset.sample_info_list)
                if location in val_locs
            ]
            test_idx = [
                idx for idx, (location, _) in enumerate(self.dataset.sample_info_list)
                if location in test_locs
            ]
            print(
                f"Split by location: train={len(train_locs)} locs, "
                f"val={len(val_locs)} locs, test={len(test_locs)} locs")
        else:
            raise ValueError(f"Unsupported split_mode: {self.split_mode}")

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
