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
   - [深入分析分类性能](#深入分析分类性能)
   - [分析空间混淆关系](#分析空间混淆关系)
6. [可视化分析](#可视化分析)
   - [位置类别分布](#位置类别分布)
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
# pip install csikit # 如果需要处理原始 .dat 文件，csikit 似乎是特定工具，请确保其可用性或替换为项目中 data_process.py 的逻辑
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
项目提供了三种基于深度学习的室内定位模型，均在 `PyTorch Lightning` 框架下实现。

### CNN_Net
-   **文件**: `cnn_net_model.py`
-   **架构**: 纯卷积神经网络。
    -   包含三层2D卷积层 (`nn.Conv2d`)，用于提取CSI数据的空间特征。
    -   使用批归一化 (`nn.BatchNorm2d` 和 `nn.BatchNorm1d`) 提高训练稳定性。
    -   最后通过全连接层 (`nn.Linear`) 进行分类。
-   **特点**: 结构相对简单，训练速度较快，适合作为基线模型。

### CNN_LSTM_Net
-   **文件**: `cnn_lstm_net_model.py`
-   **架构**: 卷积神经网络与长短期记忆网络 (LSTM) 的结合。
    -   首先通过三层2D卷积层提取空间特征，与 `CNN_Net` 类似。
    -   卷积层输出的特征图被重塑并输入到LSTM层 (`nn.LSTM`)，用于捕捉CSI数据的时间序列特性。
    -   LSTM的最后一个时间步的输出用于分类。
-   **特点**: 能够同时学习CSI数据的空间和时间依赖性，通常在高精度定位任务中表现更优。

### CNN_Transformer_Net
-   **文件**: `cnn_transformer_model.py`
-   **架构**: 卷积神经网络与Transformer编码器的结合。
    -   使用两层2D卷积层（包含最大池化 `nn.MaxPool2d`）进行初步特征提取和降维。
    -   卷积输出被视为一个序列，输入到Transformer编码器层 (`nn.TransformerEncoderLayer`, `nn.TransformerEncoder`)。Transformer利用自注意力机制捕捉序列内各元素间的复杂关系。
    -   Transformer编码器的输出用于最终分类。
-   **特点**: 强大的序列建模能力，可能在捕捉CSI数据中更长距离或更复杂的依赖关系方面具有优势。需要仔细调整超参数（如 `d_model`, `nhead`）。

所有模型均将室内定位视为一个**分类问题**，其中每个唯一的 `(x,y)` 坐标对被视为一个独立的类别。

## 模型训练

### 基础训练命令
选择并训练特定模型，例如 `CNN_LSTM_Net`:
```powershell
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train
```
将 `--model_type` 替换为 `cnn` 或 `cnn_transformer` 来训练其他模型。

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
-   形如 `<model_type>-best-epoch=XX-val_loss=Y.YYY.ckpt` (或基于 `val_acc`): 保存验证集上性能最佳的模型状态。

这些检查点可以通过 `--cpt_path` 参数加载，用于继续训练或进行测试评估。

### 常用参数说明
-   `--model_type`: (字符串) 选择模型架构。可选: `'cnn'`, `'cnn_lstm'`, `'cnn_transformer'`。默认为 `'cnn_lstm'`。
-   `--data_dir`: (字符串) 包含CSV格式CSI数据的数据集目录。
-   `--batch_size`: (整数) 训练和评估时的批处理大小。默认为 `128`。
-   `--lr`: (浮点数) 优化器的初始学习率。默认为 `0.0001`。
-   `--max_epochs`: (整数) 最大训练周期数。默认为 `120`。
-   `--min_epochs`: (整数) 最小训练周期数。默认为 `1`。
-   `--time_step`: (整数) 每个CSI样本的时间步长（即用多少个连续的CSI测量来构成一个输入样本）。默认为 `15`。
-   `--stripe` (应为 `--stride`): (整数) 在生成样本时，连续样本之间的滑动步长。默认为 `1`。
-   `--num_workers`: (整数) DataLoader使用的工作进程数。默认为 `4`。
-   `--fast_dev_run`: (布尔值) 如果为True，则运行一个批次的训练和验证，用于快速调试。默认为 `False`。
-   `--mode`: (字符串) 运行模式。可选: `'train'`, `'test'`。默认为 `'train'`。
-   `--cpt_path`: (字符串) 模型检查点文件的路径。用于从特定检查点继续训练或进行测试。

### 高级训练选项
您可以组合使用上述参数进行更细致的训练控制。例如，使用较小的批处理大小和较高的学习率训练 `CNN_Transformer_Net`：
```powershell
python main.py --model_type cnn_transformer --data_dir ./dataset --batch_size 64 --lr 0.0005 --max_epochs 200 --time_step 20 --stride 2 --mode train
```
*注意: `CNN_Transformer_Net` 可能对超参数如 `d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward` 敏感，这些目前在 `cnn_transformer_model.py` 中有默认值，高级用户可以修改模型文件以调整它们。*

## 性能评估

### 测试已训练模型
使用 `--mode test` 和 `--cpt_path` 参数测试已训练的模型在测试集上的性能：
```powershell
python main.py --mode test --model_type cnn_lstm --data_dir ./dataset --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt
```
将 `--model_type` 和 `--cpt_path` 替换为您要测试的实际模型和检查点。

### 测试不同检查点的模型
您可以通过更改 `--cpt_path` 参数来测试在不同训练阶段保存的模型，例如测试验证集上表现最佳的模型：
```powershell
# 示例：测试 CNN_LSTM 模型的最佳检查点 (文件名可能因您的训练结果而异)
python main.py --mode test --model_type cnn_lstm --data_dir ./dataset --cpt_path ./logs/cnn_lstm/version_0/checkpoints/cnn_lstm-best-epoch=XX-val_loss=Y.YYY.ckpt
```

### 深入分析分类性能
使用 `visualize_classification.py` 脚本来生成混淆矩阵和准确率热图，从而更深入地分析模型的分类性能。
```powershell
python visualize_classification.py --model_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt --model_type cnn_lstm --data_dir ./dataset
```
-   `--model_path`: 指向要评估的模型检查点文件。
-   `--model_type`: 必须与加载的模型类型一致。
-   `--data_dir`: 数据集目录。

### 分析空间混淆关系
使用 `analyze_spatial_confusion.py` 脚本来分析模型的预测错误与真实位置和预测位置之间的物理（欧几里得）距离的关系。
```powershell
python analyze_spatial_confusion.py --model_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt --model_type cnn_lstm --data_dir ./dataset
```
这有助于理解模型是否倾向于将邻近的位置混淆。

## 可视化分析

### 位置类别分布
使用 `visualize_locations.py` 脚本来可视化数据集中定义的位置点的空间分布。
```powershell
python visualize_locations.py --data_dir ./dataset
```
这有助于检查位置标签是否正确，以及采样点的覆盖范围。

## 常见问题解答

### Q: 如何选择合适的模型?
A:
-   **CNN_Net**: 适合作为快速基线或计算资源非常有限的情况。
-   **CNN_LSTM_Net**: 通常在CSI这类具有时空特性的数据上表现良好，推荐作为首选尝试。
-   **CNN_Transformer_Net**: 具有强大的建模能力，可能在复杂场景或需要捕捉更长依赖时有优势，但可能需要更多数据和更仔细的调参。

### Q: 如何处理过拟合问题?
A:
1.  **数据增强**: 虽然本项目未直接实现，但可以考虑对CSI信号进行微小的变换。
2.  **正则化**: 模型中已使用批归一化 (BatchNorm)。可以考虑在全连接层添加 Dropout 或 L2 正则化。
3.  **早停 (EarlyStopping)**: 已通过 PyTorch Lightning Callbacks 实现，监控 `val_loss`。
4.  **减小模型复杂度**: 尝试减少卷积层数、通道数，或减小LSTM/Transformer的隐藏单元数。
5.  **增加数据量**: 收集更多样化的数据。

### Q: 为什么选择分类而非回归来定位?
A: 本项目采用的是**指纹定位 (Fingerprinting)** 的思想。
-   **指纹定位**: 将每个预定义的可区分位置点（例如网格点 `(x,y)`）视为一个单独的类别。模型学习将观察到的CSI模式映射到这些预定义的位置类别之一。
-   **优点**:
    -   可以更好地处理CSI信号的复杂非线性特性。
    -   对于离散的、定义明确的参考点集合，分类方法通常更直接且易于训练。
-   **与回归对比**: 直接回归坐标 `(x,y)` 也是一种方法，但可能对CSI信号的微小变化更敏感，且损失函数的定义和优化可能更具挑战性。对于指纹库方法，分类是自然的选择。

### Q: 如何改进模型性能?
A:
1.  **数据质量与数量**:
    -   确保CSI数据质量高，噪声低。
    -   增加训练数据量，覆盖更多样化的环境条件和位置。特别是在模型易混淆的位置点增加样本。
2.  **特征工程**:
    -   探索除了幅度和相位之外的其他CSI衍生特征。
    -   尝试不同的归一化或滤波方法。
3.  **模型架构调整**:
    -   对于CNN部分，尝试不同的卷积核大小、步长、层数。
    -   对于LSTM/Transformer，调整隐藏单元数、层数、注意力头数等。
    -   考虑引入注意力机制到CNN或LSTM结构中。
4.  **超参数优化**: 系统地搜索最佳的学习率、批大小、优化器参数等。
5.  **多天线数据融合策略**: 当前是将多天线数据作为独立通道堆叠。可以研究更复杂的融合方法。
6.  **后处理**: 对模型的分类输出进行平滑处理或结合运动模型。
