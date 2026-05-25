# 新 `.mat` 数据训练方案

本文档用于指导项目基于新的 MATLAB v7.3 `.mat` 数据格式实现完整训练流程。
项目只面向 2 天线、100 子载波的 `.mat` 数据。

整体思路沿用之前的训练流程：数据预处理后缓存到内存，按滑动窗口生成
`(channels, time_step, num_subcarriers)` 样本，通过 `CSIDataModule` 完成
训练、验证、测试拆分，再用 CNN、CNN-LSTM、CNN-Transformer 进行坐标回归。

## 数据格式

| 维度 | 新数据 |
|------|--------|
| 数据来源 | 1 个 `.mat` 文件 / 位置 |
| 输入通道 | 4 (2 天线 × 幅度/相位 2 类特征) |
| 子载波数 | 100 |
| 每位置数据包 | 1000 |
| 样本形状 | `(4, time_step, 100)` |
| 回归目标 | `[x, y]` float32 坐标 |

核心字段：

- `csiAmplitudeFiltered`: shape 为 `(2, 100, 1000)`
- `csiPhaseCalibrated`: shape 为 `(2, 100, 1000)`
- `coord/x`、`coord/y`: 位置坐标

## 需要修改的文件

### 1. `csi_dataset.py` — 实现新数据加载和预处理

- 新增 `h5py` 依赖，用于读取 MATLAB v7.3 / HDF5 `.mat` 文件
- `CSIDataset.__init__`: 改为扫描 `CSI_RX_*.mat` 文件
- 坐标来源：
  - 优先读取 `.mat` 内部的 `coord/x`、`coord/y`
  - 同时用正则解析文件名中的 `x_p1d000_y_p3d000` 作为一致性校验
  - 坐标解析函数需要支持正数、负数和小数，例如 `p1d000 -> 1.0`，
    `n0d500 -> -0.5`
- `_load_all_data`: 用 `h5py` 读取每个 `.mat` 文件：
  - 读 `csiAmplitudeFiltered` (2, 100, 1000) → 转置为 (1000, 100)/天线
  - 读 `csiPhaseCalibrated` (2, 100, 1000) → 转置为 (1000, 100)/天线
  - 校验幅度和相位 shape 一致，且都是 `(n_antennas, n_subcarriers, n_packets)`
  - 根据实际数据自动设置 `num_antennas`、`num_subcarriers`、`in_channels`
  - 新数据已经过滤波处理，跳过 `median_filter`，只做 `min_max_normalization`
  - 存入 `data_cache`: 每个位置 2 个天线 × (amp, phase) 元组
- `__getitem__`: 输出 shape 为 `(4, time_step, 100)`
- 新增 `in_channels` 属性 (= 2 × num_antennas)，传递给模型
- `CSIDataModule`: 暴露 `in_channels`，与现有 `num_subcarriers` 一样供模型和
  评估脚本使用
- 保持之前的滑动窗口思路：每个位置按 `time_step` 和 `stride` 生成样本索引
- 保持之前的拆分思路：默认 `by_location`，也保留 `random` 作为快速调试选项

### 2. `cnn_net_model.py` — 输入通道参数化

- 第一层卷积使用 `Conv2d(in_channels, 18, ...)`
- `__init__` 新增 `in_channels` 参数，默认 4

### 3. `cnn_lstm_net_model.py` — 同上

- 第一层卷积使用 `Conv2d(in_channels, 18, ...)`
- `__init__` 新增 `in_channels` 参数，默认 4

### 4. `cnn_transformer_model.py` — 同上

- 第一层卷积使用 `Conv2d(in_channels, 16, ...)`
- 需要验证 `d_model = 32 * (100 // 4) = 800`，`nhead=4` 可整除，OK
- `__init__` 新增 `in_channels` 参数，默认 4

### 5. `main.py` — 传递参数

- 从 `CSIDataModule` 获取 `in_channels` 和 `num_subcarriers`，传入模型构造
- `num_subcarriers` 默认值不再硬编码，由数据自动推断（已有此逻辑）
- `test` 模式加载 checkpoint 时也要传 `in_channels`，避免新 checkpoint
  评估时构造出错误输入通道数的模型

### 6. 评估和辅助脚本 — 同步输入通道

- `visualize_classification.py`: 从 `CSIDataModule` 读取 `in_channels`，
  加入 `load_from_checkpoint` 的 `load_kwargs`
- `analyze_spatial_confusion.py`: 同上
- `simple_test.py`: 构造测试输入时从 checkpoint hyperparameters 读取
  `in_channels`，默认 4
- `visualize_locations.py`: 依赖 `CSIDataset` 自动扫描 `.mat` 后无需额外改
  调用方式，但要确认输出坐标仍正确
- `heatmappic.py`: 读取 `.mat` 中的 `csiAmplitudeFiltered` 和
  `csiPhaseCalibrated`，输出每个天线的幅度/相位热力图

### 7. 文档 — 同步新数据格式

- `CLAUDE.md`: 输入格式描述改为 4 通道，数据加载说明改为 `.mat` 文件
- `readme.md`: 更新数据集格式、通道数、依赖安装命令和文件说明
- `USAGE_GUIDE.md`: 更新数据准备、输入通道说明、依赖安装命令
- `dataset/DATA_FORMAT.md`: 统一输入张量说明为
  `(batch, 4, time_step, num_subcarriers)`

## 数据流示意

```
.mat 文件 (per location)
  csiAmplitudeFiltered: (2, 100, 1000)  →  转置  →  ant0_amp: (1000, 100)
  csiPhaseCalibrated:   (2, 100, 1000)  →  转置  →  ant0_phase: (1000, 100)
                                                      ant1_amp: (1000, 100)
                                                      ant1_phase: (1000, 100)
  ↓ min_max_normalization
  ↓ sliding window (time_step=15, stride=2)

  output sample: (4, 15, 100) = (channels, time_step, subcarriers)
  output target: [x, y] float32
```

## 端到端流程

1. 扫描 `dataset/CSI_RX_*.mat`，发现所有位置文件
2. 读取 `coord/x`、`coord/y`，并用文件名坐标做一致性校验
3. 读取 `csiAmplitudeFiltered` 和 `csiPhaseCalibrated`
4. 按天线拆分，转置为 `(n_packets, n_subcarriers)`
5. 对每个天线的幅度和相位分别做 `min_max_normalization`
6. 将 `[ant0_amp, ant0_phase, ant1_amp, ant1_phase]` 作为 4 个输入通道
7. 使用滑动窗口生成 `(4, time_step, 100)` 样本
8. 通过 `CSIDataModule` 拆分 train/val/test
9. 训练模型并保存 checkpoint
10. 使用 test 和可视化脚本输出距离误差、CDF、散点图和空间误差分析

## 实现注意事项

- `.mat` 文件中的核心数组实际 shape 是 `(2, 100, 1000)`，读取单个天线后
  需要从 `(100, 1000)` 转置为 `(1000, 100)`，即
  `(n_packets, n_subcarriers)`
- `min_max_normalization` 目前会对输入矩阵做转置再归一化，需要确认
  归一化维度是否仍符合预期；如果要保持“每个数据包内部跨子载波归一化”，
  当前 `(n_packets, n_subcarriers)` 输入是匹配的
- checkpoint 从 `.mat` 数据训练产生，模型输入形状按 4 通道设计

## 验证命令

实现完成后至少运行以下命令：

```powershell
python main.py --model_type cnn --data_dir ./dataset --mode train --fast_dev_run --num_workers 0
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train --fast_dev_run --num_workers 0
python main.py --model_type cnn_transformer --data_dir ./dataset --mode train --fast_dev_run --num_workers 0
python simple_test.py
python visualize_locations.py --data_dir ./dataset
```

如果已有新格式训练出的 checkpoint，再补充运行：

```powershell
python main.py --model_type cnn --data_dir ./dataset --mode test --num_workers 0 --cpt_path ./logs/cnn/version_0/checkpoints/last.ckpt
python visualize_classification.py --model_path ./logs/cnn/version_0/checkpoints/last.ckpt --model_type cnn --data_dir ./dataset --num_workers 0
python analyze_spatial_confusion.py --model_path ./logs/cnn/version_0/checkpoints/last.ckpt --model_type cnn --data_dir ./dataset --num_workers 0
```

## 不需要修改的文件

- **`base_model.py`**: 回归头和训练逻辑不涉及输入维度，无需改动
- **`util.py`**: `min_max_normalization` 对 `(nPackets, nSubcarriers)` 2D
  数组仍然适用；新数据已自带滤波，不再调用 `median_filter`
