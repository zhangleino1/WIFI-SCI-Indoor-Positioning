# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WiFi CSI (Channel State Information) based indoor positioning system using deep learning. The project implements multiple neural network architectures (CNN, CNN+LSTM, CNN+Transformer) to regress CSI data patterns directly to indoor `(x, y)` coordinates.

## Core Architecture

### Data Pipeline
- **Input**: 4-channel CSI data from 2 antennas (amplitude + phase per antenna)
- **Format**: `(batch_size, 4, time_step, num_subcarriers)` where 4 = 2 antennas × 2 features
- **Dataset**: MATLAB v7.3 `.mat` files named `CSI_RX_x_p{X}d{DEC}_y_p{Y}d{DEC}_z_p{Z}d{DEC}_{DATE}_{TIME}.mat`
- **Processing**: Filtered amplitude and calibrated phase are loaded into memory, normalized, and split into time-windowed segments

### Model Architectures
1. **CNN_Net** (`cnn_net_model.py`): Pure CNN with 3 conv layers + fully connected layers
2. **CNN_LSTM_Net** (`cnn_lstm_net_model.py`): CNN feature extraction + LSTM for temporal modeling
3. **CNN_Transformer_Net** (`cnn_transformer_model.py`): CNN + Transformer encoder for sequence modeling

### Key Components
- **CSIDataset** (`csi_dataset.py`): Custom PyTorch Dataset that loads CSI data and returns regression targets as `(x, y)` tensors
- **CSIDataModule** (`csi_dataset.py`): PyTorch Lightning DataModule handling train/val/test splits
- **Utility functions** (`util.py`): Data preprocessing (min-max normalization)

## Development Commands

### Training Models
```bash
# Train CNN model
python main.py --model_type cnn --data_dir ./dataset --mode train --batch_size 64 --max_epochs 120

# Train CNN+LSTM model (recommended)
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train --batch_size 64 --max_epochs 120

# Train CNN+Transformer model
python main.py --model_type cnn_transformer --data_dir ./dataset --mode train --batch_size 64 --max_epochs 120
```

### Testing Models
```bash
# Test trained model
python main.py --mode test --model_type cnn_lstm --cpt_path ./logs/cnn_lstm/version_X/checkpoints/best_checkpoint.ckpt --data_dir ./dataset
```

### Analysis and Visualization
```bash
# Visualize dataset locations
python visualize_locations.py --data_dir ./dataset

# Generate regression evaluation plots
python visualize_classification.py --model_path ./logs/cnn_lstm/version_X/checkpoints/best_checkpoint.ckpt --model_type cnn_lstm --data_dir ./dataset

# Analyze spatial regression error patterns
python analyze_spatial_confusion.py --model_path ./logs/cnn_lstm/version_X/checkpoints/best_checkpoint.ckpt --model_type cnn_lstm --data_dir ./dataset

# Generate CSI heatmap visualization
python heatmappic.py --data_dir ./dataset
```

## Key Parameters

### Model Configuration
- `--model_type`: Choose from `cnn`, `cnn_lstm`, `cnn_transformer`
- `--time_step`: Number of consecutive CSI measurements per sample (default: 15)
- `--stride`: Step size when creating overlapping time windows (default: 2)

### Training Configuration
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--max_epochs`: Maximum training epochs (default: 120)
- `--min_epochs`: Minimum training epochs (default: 10)

### Data Configuration
- `--data_dir`: Path to dataset directory containing `.mat` files (default: ./dataset)
- `--num_workers`: DataLoader worker processes (default: 8)

## Important Implementation Details

### Regression Approach
- Each sample predicts a continuous `(x, y)` coordinate
- File names still define the target location for each CSI window
- Models output coordinates directly, not class probabilities

### Data Loading Strategy
- All `.mat` CSI arrays are pre-loaded into memory for faster training
- Data is cached with location keys for efficient access
- Default `by_location` split keeps locations disjoint across train/validation/test
- `by_location` is a stricter generalization test, but small datasets may yield higher-variance metrics

### Model Checkpoints
- Best models saved based on validation loss in `./logs/<model_type>/version_N/checkpoints/`
- Use `last.ckpt` for most recent checkpoint or `*-best-*.ckpt` for best validation performance

### Framework
- Built on PyTorch Lightning for simplified training/validation loops
- Automatic GPU detection and usage
- Integrated TensorBoard logging
- Early stopping and learning rate scheduling included

## Environment Requirements

The project requires:
- Python 3.8+
- PyTorch with CUDA support (if GPU available)
- PyTorch Lightning
- scikit-learn, pandas, numpy, h5py
- matplotlib, seaborn (for visualizations)

Install via conda/pip as specified in USAGE_GUIDE.md.

## Data Format Requirements

`.mat` files must follow naming convention:
`CSI_RX_x_p{X}d{DEC}_y_p{Y}d{DEC}_z_p{Z}d{DEC}_{DATE}_{TIME}.mat`
- Each file contains `csiAmplitudeFiltered` and `csiPhaseCalibrated` arrays with shape `(2, 100, 1000)`.
- The loader uses `coord/x` and `coord/y` as the regression target and validates them against the file name.
- Each model input sample has shape `(4, time_step, 100)`.

## Testing and Validation

- Models automatically report regression distance metrics during testing
- Use visualization scripts to analyze model performance and spatial error patterns
- Check logs directory for TensorBoard training metrics
- Validation loss is primary metric for model selection (early stopping)
