**Project Name:** 
WIFI-SCI-Indoor-Positioning


**Description:**
WIFI-SCI-Indoor-Positioning is an innovative open-source project designed to leverage deep learning techniques for precise indoor positioning using WiFi Channel State Information (CSI). This project taps into the rich data provided by CSI, which includes information about the signal's amplitude, phase, and the propagation environment, to accurately estimate device locations inside buildings where GPS signals are unreliable.

**Objective:**
Our objective is to develop and refine models that can interpret complex patterns in CSI data, allowing for the accurate detection of a device's location based on the environmental characteristics reflected in the WiFi signals.

**How It Works:**
- **Data Collection:** We collect CSI data from WiFi networks, capturing details that traditional RSSI-based systems overlook.
- **Model Training:** We use various deep learning architectures, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), to learn from the temporal and spatial characteristics of the CSI data.
- **Position Estimation:** The trained models predict the location by interpreting the processed CSI data, considering factors like signal multipath effects, fading, and shadowing.

**Key Features:**
- **High Accuracy:** By utilizing deep learning, the project aims to significantly improve the accuracy of indoor positioning compared to traditional methods.
- **Scalability:** Designed to work seamlessly with existing WiFi infrastructure, making it easily scalable across different environments.
- **Real-Time Processing:** Capable of processing live data to provide real-time location services.
- **Community-Driven:** Encourages contributions from researchers and developers to further enhance the functionality and adaptability of the solution.

**Potential Applications:**
- Enhancing navigation systems in complex indoor environments such as airports, malls, and hospitals.
- Enabling location-based services and advertisements.
- Improving safety and security monitoring by providing precise location tracking within indoor spaces.

Below is the translated Chinese version of your project description:

---

## 项目名称：
WIFI-SCI-室内定位

---

## 项目描述：
**WIFI-SCI-室内定位**是一个创新的开源项目，旨在利用深度学习技术，通过WiFi信道状态信息（CSI）实现精准的室内定位。本项目充分利用CSI提供的丰富数据，包括信号的幅度、相位以及传播环境信息，从而在GPS信号不可靠的室内环境中准确估算设备位置。

---

## 项目目标：
我们的目标是开发和完善模型，使其能够解读CSI数据中的复杂模式，从而根据WiFi信号反映的环境特征准确检测设备的位置。

---

## 工作原理：
* **数据收集：** 我们从WiFi网络收集CSI数据，捕捉传统基于RSSI（接收信号强度指示）的系统所忽略的细节。
* **模型训练：** 我们使用各种深度学习架构，如卷积神经网络（CNN）和循环神经网络（RNN），从CSI数据的时间和空间特征中进行学习。
* **位置估算：** 经过训练的模型通过解释处理后的CSI数据来预测位置，同时考虑信号多径效应、衰落和阴影等因素。

---

## 主要特点：
* **高精度：** 通过利用深度学习，本项目旨在显著提高室内定位的精度，超越传统方法。
* **可扩展性：** 旨在与现有WiFi基础设施无缝协作，使其在不同环境中易于扩展。
* **实时处理：** 能够处理实时数据以提供实时定位服务。
* **社区驱动：** 鼓励研究人员和开发者的贡献，以进一步增强解决方案的功能和适应性。

---

## 潜在应用：
* 增强机场、商场和医院等复杂室内环境的导航系统。
* 实现基于位置的服务和广告。
* 通过在室内空间提供精准的位置跟踪，提高安全监控水平。

# 复现论文
csi.pdf 仔细看在项目中

# 室内定位算法定制
如果你也想定制自己的室内定位算法，请联系我！

喝杯咖啡，感谢开源，祝您毕业！

![dashang](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/dashang.png)

![公众号](https://img-blog.csdnimg.cn/9996d8ee490a402aaa7243ba84aef175.png)

# 数据集
csv 文件 格式说明antenna_1_2_6.csv  antenna_天线号_坐标x_坐标y.csv

[全量csv数据集](https://pan.quark.cn/s/be9b44dd75b6)
[全量dat原始数据集](https://pan.quark.cn/s/b2349706d0f6)

注意：`dataset`目录中提供了少量数据集用于快速测试和演示，并非完整数据集。

# 文件说明
本项目基于 PyTorch Lightning 构建，主要文件包括：
```
main.py                     # 训练、测试和预测的统一入口脚本
csi_dataset.py              # PyTorch Dataset 和 DataLoader 实现，负责数据加载和预处理
cnn_net_model.py            # 纯 CNN 模型架构
cnn_lstm_net_model.py       # CNN + LSTM 混合模型架构
cnn_transformer_model.py    # CNN + Transformer 混合模型架构
util.py                     # 数据预处理工具函数 (如归一化、中值滤波)
data_process.py             # (可选) 原始 .dat 文件到 .csv 文件的转换脚本
heatmappic.py               # (可选) CSI 数据热力图生成，用于数据质量初步检查
visualize_locations.py      # 可视化数据集中位置点的分布
visualize_classification.py # 可视化模型分类结果 (混淆矩阵、准确率热图)
analyze_spatial_confusion.py # 分析模型预测错误与物理空间距离的关系
USAGE_GUIDE.md              # 详细的用户使用指南
readme.md                   # 本项目概览文件 (您正在阅读的文件)
csi.pdf                     # (参考) 项目可能复现或参考的相关论文
```

# Python 实现说明

## 数据集处理 (`csi_dataset.py`)
- 采用 PyTorch 的 `Dataset` 和 `DataLoader` 机制高效处理CSI数据。
- 输入数据为 **6通道**，由3个天线的幅度和相位数据构成 (`3 antennas * (1 amplitude + 1 phase) = 6 channels`)。
- 使用 `OrderedDict` 实现数据缓存机制，加速重复数据块的加载，提高训练效率。
- 将室内定位问题从坐标回归转换为**分类问题**，每个唯一的 `(x,y)` 坐标被视为一个独立的类别。
- 自动扫描数据集目录，提取所有唯一位置点，并创建位置到类别索引 (`location_to_class`) 和类别索引到位置 (`class_to_location`) 的映射。

## 网络模型
本项目提供了三种基于 PyTorch Lightning 实现的深度学习模型：

### 1. CNN 模型 (`cnn_net_model.py`)
-   **架构**: 纯卷积神经网络。包含三层2D卷积层 (`nn.Conv2d`)，主要用于提取CSI数据中的空间特征。
-   **特点**: 结构相对简单，训练速度快，适合作为基线模型或在计算资源受限时使用。

### 2. CNN + LSTM 模型 (`cnn_lstm_net_model.py`)
-   **架构**: 结合了CNN和长短期记忆网络 (LSTM)。首先通过CNN提取空间特征，然后将这些特征序列输入到LSTM层 (`nn.LSTM`) 以捕捉数据的时间依赖性。
-   **特点**: 能够同时学习CSI数据的空间和时间特性，通常在处理具有时序性的CSI数据时表现更优，是本项目推荐尝试的主力模型之一。

### 3. CNN + Transformer 模型 (`cnn_transformer_model.py`)
-   **架构**: 结合了CNN和Transformer编码器。通过CNN进行初步特征提取和降维，然后将输出序列输入Transformer编码器 (`nn.TransformerEncoder`)，利用其自注意力机制学习序列内的复杂关系。
-   **特点**: Transformer具有强大的序列建模能力，可能在捕捉CSI数据中更长距离或更复杂的依赖关系方面有优势。对超参数和数据量可能更敏感。

所有模型均：
- 使用 `CrossEntropyLoss` 作为损失函数，适应分类任务。 目前精度还没调好，可以改成回归任务也可以。
- 在训练、验证和测试阶段监控准确率。
- 支持生成混淆矩阵以进行详细的性能分析。

## 训练与评估 (`main.py`)
- 基于 PyTorch Lightning 实现，简化了训练循环、GPU使用、日志记录等。
- 支持通过命令行参数配置模型类型、学习率、批大小、训练周期等。
- 集成了早停 (EarlyStopping) 和学习率调度 (ReduceLROnPlateau) 等高级训练策略。
- 自动保存最佳模型检查点和最后一个检查点。

## 数据可视化
- `visualize_locations.py`: 可视化数据集中所有位置点的空间分布。
- `visualize_classification.py`: 加载已训练模型，评估其在测试集上的表现，并生成混淆矩阵和分类准确率热图。
- `analyze_spatial_confusion.py`: 分析模型预测错误的位置与真实位置之间的物理距离，帮助理解模型的混淆模式。

## 详细使用说明

有关环境配置、数据准备、模型训练、参数调整和结果分析的完整详细步骤，请参阅 **[详细使用指南 (USAGE_GUIDE.md)](./USAGE_GUIDE.md)**。

# Email
zhangleilikejay@gmail.com

# 技术细节

## 环境要求
- Python 3.8+
- PyTorch (推荐1.12+ 或更高版本，支持CUDA)
- PyTorch Lightning (推荐1.6+ 或更高版本)
- 详细依赖请参见 `USAGE_GUIDE.md` 中的环境配置部分。

## 数据格式
- 输入数据为CSV文件，命名格式: `antenna_<天线号>_<x坐标>_<y坐标>.csv`。
- 每个CSV文件包含多行CSI样本，每行代表一个时间点的测量。
- 列包含幅度和相位数据，例如: `amplitude_0`, ..., `amplitude_N-1`, `phase_0`, ..., `phase_N-1`。
- 详细说明请参见 `USAGE_GUIDE.md`。
## 指纹地图

![map](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/wechat_2025-08-21_093130_935.png?raw=true)

## 分类实现关键点
1. **类别映射**: 将每个唯一的 (x,y) 坐标作为一个分类类别
   ```python
   self.location_to_class = {loc: idx for idx, loc in enumerate(sorted(self.location_classes))}
   ```

2. **模型输出层**: 从回归输出(2个神经元)改为类别数量的神经元
   ```python
   self.fc3 = nn.Linear(900, num_classes)  # 输出类别数量的神经元
   ```

3. **损失函数**: 从 MSE 损失改为交叉熵损失
   ```python
   loss = F.cross_entropy(logits, targets)
   ```

4. **准确率评估**: 在训练、验证和测试阶段添加准确率指标
   ```python
   preds = torch.argmax(logits, dim=1)
   acc = (preds == targets).float().mean()
   ```

## 可视化分析方法
- **混淆矩阵分析**: 识别模型在哪些位置类别之间容易产生混淆。
- **位置准确率热图**: 直观展示模型在不同物理位置上的预测准确率。
- **物理距离误差分析**: 分析错误分类的样本与其真实位置和预测位置之间的物理（欧几里得）距离。

# 报错 注意安装
```powershell
# 强烈建议参照 USAGE_GUIDE.md 中的详细环境配置步骤进行安装
# 关键依赖:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia (根据您的CUDA版本调整)
# conda install lightning -c conda-forge
# pip install scikit-learn matplotlib seaborn pandas tensorboard
```

# 可能的未来优化方向
- 添加多传感器融合功能，结合其他定位数据源
- 开发在线学习模块，使系统能够适应环境变化
- 增加少样本学习能力，减少每个位置需要的训练数据量

# csi data image filter

![1](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/1.png)
![2](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/2.png)
![3](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/3.png)
![4](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/4.png)

# research images
![1](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/Figure_1.png)
![2](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/cnn_cdf.png)
![3](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/ambiguous_locations.png)
![4](https://cdn.nlark.com/yuque/0/2024/png/354158/1721299909950-46a16f6b-cbd8-40fb-9833-35be7e0d0c5c.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_1303%2Climit_0)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zhangleino1/WIFI-SCI-Indoor-Positioning&type=Date)](https://star-history.com/#zhangleino1/WIFI-SCI-Indoor-Positioning&Date)
