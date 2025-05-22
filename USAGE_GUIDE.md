# WIFI-CSI 室内定位系统使用指南

本文档提供了有关如何使用 WIFI-SCI-Indoor-Positioning 系统的详细说明，包括环境配置、数据处理、模型训练和结果分析等方面。

## 目录
1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [性能评估](#性能评估)
5. [可视化分析](#可视化分析)
6. [常见问题解答](#常见问题解答)

## 环境配置

### 创建虚拟环境
```powershell
conda create -n csi-positioning python=3.8
conda activate csi-positioning
```

### 安装依赖包
```powershell
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install lightning -c conda-forge
pip install -U 'tensorboardX'
pip install -U 'tensorboard'
pip install scikit-learn matplotlib seaborn pandas
pip install csikit
```

### 检查CUDA是否可用
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA current device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
```

## 数据准备

### 数据格式
项目使用的CSV数据文件应遵循以下命名格式：
```
antenna_<天线号>_<x坐标>_<y坐标>.csv
```

例如：`antenna_1_3_5.csv` 表示天线1在坐标(3,5)处采集的CSI数据。

每个CSV文件包含幅度和相位数据，列命名格式为：
- 幅度列: `amplitude_0`, `amplitude_1`, ...
- 相位列: `phase_0`, `phase_1`, ...

### 数据处理流程
1. 使用 `data_process.py` 将原始dat文件转换为CSV
```powershell
python data_process.py --input_dir ./raw_data --output_dir ./dataset
```

2. 使用 `heatmappic.py` 生成热力图，辅助数据质量检查
```powershell
python heatmappic.py --data_dir ./dataset
```

## 模型训练

### 基础训练命令
```powershell
python main.py --model_type cnn_lstm --data_dir ./dataset --mode train
```

### 训练与测试模式说明
系统提供两种运行模式：
- **训练模式 (--mode train)**: 用于从头开始训练模型，会创建新的模型并保存到日志目录
- **测试模式 (--mode test)**: 用于加载已训练的模型并进行测试，需要配合 `--cpt_path` 使用

### 模型检查点 (Checkpoint) 说明
训练模型后，系统会自动保存检查点文件到以下路径：
```
./logs/<model_type>/version_<N>/checkpoints/
```
其中包含 `last.ckpt` (最后一个检查点) 和最佳性能的检查点。这些检查点可以通过 `--cpt_path` 参数加载进行测试或继续训练。

### 常用参数说明
- `--model_type`: 选择模型类型，可选 'cnn' 或 'cnn_lstm' 或 'cnn_transformer'，默认为 'cnn_lstm'
- `--data_dir`: 数据集目录
- `--batch_size`: 批大小，默认为128
- `--lr`: 学习率，默认为0.0001
- `--max_epochs`: 最大训练轮次，默认为120
- `--mode`: 运行模式，可选 'train' 或 'test'，默认为'train'。'train'用于训练新模型，'test'用于测试现有模型
- `--cpt_path`: 模型检查点路径，用于加载已训练的模型进行测试，默认为当前工作目录下的'/checkpoints/last.ckpt'

### 高级训练选项
```powershell
python main.py --model_type cnn_lstm --data_dir ./dataset --batch_size 64 --lr 0.0005 --max_epochs 200
```

## 性能评估

### 测试已训练模型
使用 `--mode test` 和 `--cpt_path` 参数测试已训练的模型性能：

```powershell
python main.py --mode test --model_type cnn_lstm --data_dir ./dataset --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt
```

### 测试不同检查点的模型
您可以通过更改 `--cpt_path` 参数来测试不同阶段保存的模型：

```powershell
# 测试最后保存的检查点
python main.py --mode test --model_type cnn_lstm --cpt_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt

# 测试最佳性能检查点
python main.py --mode test --model_type cnn_lstm --cpt_path ./logs/cnn_lstm/version_0/checkpoints/cnn_lstm-best-epoch=45-val_acc=0.987.ckpt
```

### 深入分析分类性能
```powershell
python visualize_classification.py --model_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt --model_type cnn_lstm --data_dir ./dataset
```

### 分析空间混淆关系
```powershell
python analyze_spatial_confusion.py --model_path ./logs/cnn_lstm/version_0/checkpoints/last.ckpt --model_type cnn_lstm --data_dir ./dataset
```

## 可视化分析

### 位置类别分布
```powershell
python visualize_locations.py --data_dir ./dataset
```


## 常见问题解答

### Q: 如何选择合适的模型?
A: 对于高精度定位，推荐使用 CNN_LSTM 模型，它能更好地捕捉时间和空间特征。如果计算资源有限，可以使用更轻量的 CNN 模型。

### Q: 如何处理过拟合问题?
A: 尝试减小网络规模、增加正则化、使用早停策略或增加数据量。当前实现已包含 BatchNorm 和早停机制。

### Q: 为什么选择分类而非回归?
A: 因为是指纹定位。

### Q: 如何改进模型性能?
A: 
1. 增加数据量，尤其是在容易混淆的位置
2. 调整网络架构，如增加注意力机制
3. 尝试不同的数据预处理方法
4. 考虑多传感器融合方法
