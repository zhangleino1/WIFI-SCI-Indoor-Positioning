# WIFI-CSI 室内定位系统使用指南

本文档提供了有关如何使用 WIFI-SCI-Indoor-Positioning 系统的详细说明，包括环境配置、数据准备、模型训练和结果分析等方面。

## 目录
1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
   - [数据格式](#数据格式)
   - [输入数据通道说明](#输入数据通道说明)
   - [数据处理流程](#数据处理流程)
3. [模型概览](#模型概览)
   - [CNN_Net](#cnn_net)
   - [CNN_LSTM_Net](#cnn_lstm_net)
   - [CNN_Transformer_Net](#cnn_transformer_net)
4. [模型训练](#模型训练)
   - [基础训练命令](#基础训练命令)
   - [训练与测试模式说明](#训练与测试模式说明)
   - [模型检查点说明](#模型检查点-checkpoint-说明)
   - [常用参数说明](#常用参数说明)
   - [高级训练选项](#高级训练选项)
5. [性能评估](#性能评估)
   - [测试已训练模型](#测试已训练模型)
   - [测试不同检查点的模型](#测试不同检查点的模型)
6. [可视化分析](#可视化分析)
7. [常见问题解答](#常见问题解答)

## 环境配置

### 创建虚拟环境
```powershell
conda create -n csi-positioning python=3.8
conda activate csi-positioning
```

### 安装依赖包
```powershell
# 根据您的 CUDA 版本选择合适的 cudatoolkit (例如 11.3, 11.6, 11.7, 11.8, 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install -U 'tensorboardX' 'tensorboard'
pip install scikit-learn matplotlib seaborn pandas
```

### 检查CUDA是否可用
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, will use CPU.")
```

## 数据准备

### 数据格式
项目使用的CSV数据文件应遵循以下命名格式：
```
antenna_<天线号>_<x坐标>_<y坐标>.csv
```

例如：`antenna_1_3_5.csv` 表示天线1在坐标(3,5)处采集的CSI数据。

每个CSV文件包含幅度和相位数据，列命名格式为：
- 幅度列: `amplitude_0`, `amplitude_1`, ... , `amplitude_N-1` (N为子载波数)
- 相位列: `phase_0`, `phase_1`, ... , `phase_N-1`

### 输入数据通道说明
模型期望的输入数据为 **6通道**。这6个通道由以下方式构成：
- **3个天线**: 系统处理来自3个不同天线的CSI数据。
- **2种特征**: 对于每个天线，提取两种信号特征：
    1.  **幅度 (Amplitude)** 数据
    2.  **相位 (Phase)** 数据

因此，数据组织方式为：
`[天线1幅度, 天线1相位, 天线2幅度, 天线2相位, 天线3幅度, 天线3相位]`
每个通道的形状为 `(time_step, num_subcarriers)`。
最终输入到模型的样本数据形状为 `(6, time_step, num_subcarriers)`。

### 数据处理流程
1.  **原始数据转换 (可选)**: 如果您有原始 `.dat` 格式的CSI数据，可以使用 `data_process.py` 将其转换为项目所需的CSV格式。
    ```powershell
    python data_process.py --input_dir ./raw_data --output_dir ./dataset
    ```
    *注意: `dataset` 目录中已提供少量样本数据用于快速测试。如需处理完整数据集，请确保 `raw_data` 目录包含您的 `.dat` 文件。*

2.  **数据质量检查 (可选)**: 使用 `heatmappic.py` 生成CSI数据的热力图，辅助进行数据质量检查。
    ```powershell
    python heatmappic.py --data_dir ./dataset
    ```
    这有助于识别异常或缺失的数据。

## 模型概览
项目提供了三种基于深度学习的室内定位回归模型，均在 `PyTorch Lightning` 框架下实现，共同继承 `base_model.py` 中的 `CSIBaseModel` 基类。所有模型直接预测 `(x, y)` 坐标，使用欧氏距离误差（单位：网格单元）衡量性能。

### CNN_Net
-   **文件**: `cnn_net_model.py`
-   **架构**: 纯卷积神经网络。
    -   包含三层2D卷积层 (`nn.Conv2d`, kernel=5, padding=2)，用于提取CSI数据的空间特征。
    -   使用批归一化 (`nn.BatchNorm2d` 和 `nn.BatchNorm1d`) 提高训练稳定性。
    -   通过全连接层 (FC 1024 → FC 512 → FC 2) 输出 `(x, y)` 坐标。
-   **特点**: 结构相对简单，训练速度较快，适合作为基线模型。

### CNN_LSTM_Net
-   **文件**: `cnn_lstm_net_model.py`
-   **架构**: 卷积神经网络与长短期记忆网络 (LSTM) 的结合。
    -   首先通过三层2D卷积层提取空间特征。
    -   卷积层输出的特征图被重塑并输入到LSTM层 (`nn.LSTM`, hidden=100)，用于捕捉CSI数据的时间序列特性。
    -   取LSTM最后一个时间步的输出，通过 FC(2) 输出 `(x, y)` 坐标。
-   **特点**: 能够同时学习CSI数据的空间和时间依赖性，通常在高精度定位任务中表现更优，**推荐作为首选模型**。

### CNN_Transformer_Net
-   **文件**: `cnn_transformer_model.py`
-   **架构**: 卷积神经网络与Transformer编码器的结合。
    -   使用两层2D卷积层（包含最大池化 `nn.MaxPool2d`）进行初步特征提取和降维。
    -   卷积输出被视为序列，输入到Transformer编码器 (`nn.TransformerEncoder`)，利用自注意力机制捕捉序列内的复杂关系。
    -   取第一个 token 的输出，通过 FC(2) 输出 `(x, y)` 坐标。
-   **特点**: 强大的序列建模能力，可能在捕捉CSI数据中更长距离或更复杂的依赖关系方面具有优势。

## 模型训练

### 基础训练命令

训练 CNN_LSTM 模型（推荐，使用默认 Smooth L1 损失）：
```powershell
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train
```

训练 CNN 模型：
```powershell
python main.py --model_type cnn --data_dir ./dataset --mode train
```

训练 CNN_Transformer 模型：
```powershell
python main.py --model_type cnn_transformer --data_dir ./dataset --mode train
```

### 训练与测试模式说明
系统提供两种运行模式，通过 `--mode` 参数指定：
-   **训练模式 (`--mode train`)**: 用于从头开始训练模型或继续之前的训练。如果指定了 `--cpt_path`，则会从该检查点继续训练；否则，将创建新模型。模型检查点会保存到日志目录。
-   **测试模式 (`--mode test`)**: 用于加载已训练的模型并进行评估。必须配合 `--cpt_path` 参数指定要加载的模型检查点。

### 模型检查点 (Checkpoint) 说明
训练过程中，PyTorch Lightning 会自动保存模型检查点。这些文件通常位于：
```
./logs/<model_type>/version_<N>/checkpoints/
```
其中：
-   `<model_type>` 是您选择的模型 (e.g., `cnn_lstm`)。
-   `<N>` 是实验的版本号。
-   `last.ckpt`: 保存最后一个训练周期结束时的模型状态。
-   形如 `<model_type>-best-epoch=XX-val_loss=Y.YYY.ckpt`: 保存验证损失最优的模型状态。

这些检查点可以通过 `--cpt_path` 参数加载，用于继续训练或进行测试评估。

### 常用参数说明
-   `--model_type`: (字符串) 选择模型架构。可选: `'cnn'`, `'cnn_lstm'`, `'cnn_transformer'`。默认为 `'cnn'`。
-   `--data_dir`: (字符串) 包含CSV格式CSI数据的数据集目录。
-   `--batch_size`: (整数) 训练和评估时的批处理大小。默认为 `64`。
-   `--lr`: (浮点数) 优化器的初始学习率。默认为 `0.001`。
-   `--max_epochs`: (整数) 最大训练周期数。默认为 `120`。
-   `--min_epochs`: (整数) 最小训练周期数。默认为 `10`。
-   `--time_step`: (整数) 每个CSI样本的时间步长（即用多少个连续的CSI测量来构成一个输入样本）。默认为 `15`。
-   `--stride`: (整数) 在生成样本时，连续样本之间的滑动步长。默认为 `2`。
-   `--num_workers`: (整数) DataLoader使用的工作进程数。默认为 `8`。
-   `--fast_dev_run`: (布尔值) 如果为True，则运行一个批次的训练和验证，用于快速调试。默认为 `False`。
-   `--split_mode`: (字符串) 数据集划分方式。可选: `'by_location'`, `'random'`。默认为 `'by_location'`。
    - `by_location`: 按位置点划分训练/验证/测试集，避免重叠滑窗泄漏，更适合评估泛化能力。
    - `random`: 按样本窗口随机划分，结果通常更乐观，仅适合快速调试。
-   `--mode`: (字符串) 运行模式。可选: `'train'`, `'test'`。默认为 `'train'`。
-   `--reg_loss`: (字符串) 回归损失函数。可选: `'smooth_l1'`, `'mse'`, `'mae'`。默认为 `'smooth_l1'`。
    - `smooth_l1` (Huber Loss): 对离群点鲁棒，兼顾MSE和MAE的优点，**推荐默认选项**。
    - `mse`: 均方误差，对大误差的惩罚更强，适合希望严格约束大误差的场景。
    - `mae`: 平均绝对误差，对离群点最鲁棒，适合数据噪声较大的场景。
-   `--cpt_path`: (字符串) 模型检查点文件的路径。用于从特定检查点继续训练或进行测试。

### 高级训练选项
指定损失函数训练 CNN_LSTM 模型：
```powershell
# 使用 MAE 损失（对噪声更鲁棒）
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train --reg_loss mae

# 使用 MSE 损失（对大误差惩罚更强）
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train --reg_loss mse
```

训练 CNN_Transformer 模型，指定更多超参数：
```powershell
python main.py --model_type cnn_transformer --data_dir ./dataset \
  --batch_size 64 --lr 0.0005 --max_epochs 200 \
  --time_step 20 --stride 2 --reg_loss smooth_l1 --mode train
```
*注意: `CNN_Transformer_Net` 的 `nhead`, `num_encoder_layers`, `dim_feedforward` 参数目前在 `cnn_transformer_model.py` 中有默认值，高级用户可以修改模型文件进行调整。*

从检查点继续训练：
```powershell
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train \
  --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt
```

## 性能评估

### 测试已训练模型

测试回归模型，输出平均/中位距离误差、命中率及可视化图表：
```powershell
python main.py --mode test --model_type cnn_lstm --data_dir ./dataset \
  --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt
```

测试结果包括：
- **平均欧氏距离误差** (mean Euclidean distance error，单位：网格单元)
- **中位欧氏距离误差** (median error)
- **命中率** (within 1 / 2 / 3 网格单元的样本占比)
- **误差CDF图**: `{model_name}_regression_cdf.png`（保存到 `results/{model_name}_results/`）
- **预测散点图**: `{model_name}_regression_scatter.png`（真实坐标 vs 预测坐标）

### 测试不同检查点的模型
您可以通过更改 `--cpt_path` 参数来测试在不同训练阶段保存的模型，例如测试验证集上表现最佳的模型：
```powershell
# 示例：测试 CNN_LSTM 模型的最佳检查点 (文件名可能因您的训练结果而异)
python main.py --mode test --model_type cnn_lstm --data_dir ./dataset \
  --cpt_path ./logs/cnn_lstm/version_0/checkpoints/cnn_lstm-best-epoch=XX-val_loss=Y.YYY.ckpt
```

## 可视化分析

### 位置分布可视化
使用 `visualize_locations.py` 脚本来可视化数据集中定义的位置点的空间分布：
```powershell
python visualize_locations.py --data_dir ./dataset
```
这有助于检查位置标签是否正确，以及采样点的覆盖范围。

## 常见问题解答

### Q: 如何选择合适的模型?
A:
-   **CNN_Net**: 适合作为快速基线或计算资源非常有限的情况。
-   **CNN_LSTM_Net**: 通常在CSI这类具有时空特性的数据上表现良好，**推荐作为首选**。
-   **CNN_Transformer_Net**: 具有强大的建模能力，可能在复杂场景或需要捕捉更长依赖时有优势，但可能需要更多数据和更仔细的调参。

### Q: 如何选择合适的损失函数 (`--reg_loss`)?
A:
-   **`smooth_l1`** (默认，推荐): Huber Loss，综合了 MSE 和 MAE 的优点。对小误差采用 L2 惩罚，对大误差（离群点）采用 L1 惩罚，鲁棒性好。
-   **`mse`**: 均方误差。当您希望模型更严格地约束大误差时选用，但离群点对训练影响较大。
-   **`mae`**: 平均绝对误差。当数据中存在较多噪声或离群点时选用，收敛速度可能稍慢。

### Q: 如何处理过拟合问题?
A:
1.  **早停 (EarlyStopping)**: 已通过 PyTorch Lightning Callbacks 实现，监控 `val_loss`。
2.  **学习率调度**: 使用 ReduceLROnPlateau，验证集 `val_loss` 不下降时自动降低学习率。
3.  **正则化**: 模型中已使用批归一化 (BatchNorm) 和 Dropout。
4.  **减小模型复杂度**: 尝试减少卷积层数、通道数，或减小LSTM/Transformer的隐藏单元数。
5.  **增加数据量**: 收集更多样化的数据。

### Q: `by_location` 划分有什么局限?
A:
-   该模式会把测试集位置点完全从训练集中隔离，更接近真实泛化测试。
-   如果位置点总数较少，测试集可能只包含少量位置，指标方差会偏大。
-   随机划分位置时，训练集和测试集的空间覆盖可能不均匀；解读结果时要结合位置分布图一起看。

### Q: 如何改进定位精度?
A:
1.  **数据质量与数量**:
    -   确保CSI数据质量高，噪声低。
    -   增加训练数据量，覆盖更多样化的环境条件和位置。
2.  **损失函数调整**: 尝试不同的 `--reg_loss` 选项，观察对误差分布的影响。
3.  **时间步长调整**: 增大 `--time_step` 可让模型利用更多历史信息；减小 `--stride` 可增加训练样本数量（但会增加相邻样本间的重叠度）。
4.  **特征工程**:
    -   探索除了幅度和相位之外的其他CSI衍生特征。
    -   尝试不同的归一化或滤波方法。
5.  **模型架构调整**:
    -   对于CNN部分，尝试不同的卷积核大小、步长、层数。
    -   对于LSTM，调整隐藏单元数和层数。
    -   对于Transformer，调整 `nhead`, `num_encoder_layers`, `dim_feedforward`。
6.  **超参数优化**: 系统地搜索最佳的学习率、批大小等。

### Q: 回归模型的误差指标如何解读?
A:
-   **平均/中位距离误差**: 单位为"网格单元"，即坐标系中的一格距离。如果您的网格间距为1米，则误差单位也为米。
-   **命中率 (within N 网格单元)**: 表示预测误差在 N 个网格单元以内的样本占总样本的百分比。within-1 命中率越高，说明精细定位能力越强。
-   **误差CDF图**: 横轴为距离误差，纵轴为累积概率。曲线越靠左上角，模型整体性能越好。
-   **预测散点图**: 横轴为真实坐标，纵轴为预测坐标，理想情况下散点沿 y=x 对角线分布。
