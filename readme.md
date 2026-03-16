**Project Name:**
WIFI-SCI-Indoor-Positioning


**Description:**
WIFI-SCI-Indoor-Positioning is an open-source project that leverages deep learning for precise indoor positioning using WiFi Channel State Information (CSI). The system directly regresses device coordinates (x, y) from CSI amplitude and phase data, enabling accurate location estimation in environments where GPS is unavailable.

**Objective:**
Develop and refine deep learning models that interpret complex patterns in CSI data to predict device coordinates in continuous 2-D space, using mean Euclidean distance error as the primary evaluation metric.

**How It Works:**
- **Data Collection:** CSI data is collected from WiFi networks across predefined grid locations.
- **Model Training:** CNN, CNN-LSTM, and CNN-Transformer architectures learn to map CSI fingerprints to (x, y) coordinates.
- **Position Estimation:** Models output continuous coordinate predictions; performance is measured by mean and median distance error (in grid units).

**Key Features:**
- **Regression-based positioning:** Directly predict (x, y) coordinates — no discretisation, no class boundaries.
- **Three model architectures:** CNN (fast baseline), CNN-LSTM (temporal modelling), CNN-Transformer (attention-based).
- **Configurable regression loss:** `smooth_l1` (default), `mse`, or `mae`.
- **Rich evaluation:** Mean / median distance error, within-N-unit hit rates, CDF plot, true-vs-predicted scatter plot.
- **Scalable:** Works with existing WiFi infrastructure; no hardware changes required.

**Potential Applications:**
- Navigation in airports, malls, and hospitals.
- Location-based services.
- Indoor asset tracking and safety monitoring.

Below is the translated Chinese version:

---

## 项目名称：
WIFI-SCI-室内定位

---

## 项目描述：
**WIFI-SCI-室内定位**是一个开源项目，利用深度学习通过 WiFi 信道状态信息（CSI）实现精准室内定位。系统直接从 CSI 的幅度和相位数据中回归预测设备坐标 (x, y)，在 GPS 信号不可用的室内环境中实现准确的位置估算。

---

## 项目目标：
开发和完善深度学习模型，将 CSI 指纹映射到连续二维坐标空间，以平均欧氏距离误差作为主要评估指标。

---

## 工作原理：
* **数据收集：** 在预定义网格位置收集 WiFi CSI 数据（幅度 + 相位）。
* **模型训练：** CNN、CNN-LSTM、CNN-Transformer 三种架构学习将 CSI 特征映射到 (x, y) 坐标。
* **位置估算：** 模型输出连续坐标预测，以平均/中位欧氏距离误差（网格单元）衡量性能。

---

## 主要特点：
* **回归定位：** 直接预测连续坐标，无需离散化，天然利用空间连续性先验。
* **三种模型架构：** CNN（快速基线）、CNN-LSTM（时序建模）、CNN-Transformer（注意力机制）。
* **可配置回归损失：** `smooth_l1`（默认）、`mse`、`mae`。
* **丰富的评估指标：** 平均/中位距离误差、N 单元命中率、误差 CDF 图、真实 vs 预测散点图。

---

## 潜在应用：
* 增强机场、商场和医院等复杂室内环境的导航系统。
* 实现基于位置的服务和广告。
* 通过室内空间精准位置跟踪提高安全监控水平。

# 复现论文
csi.pdf 仔细看在项目中

# 室内定位算法定制
如果你也想定制自己的室内定位算法，请联系我！

喝杯咖啡，感谢开源，祝您毕业！

![dashang](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/dashang.png)


![程序员石磊](https://github.com/zhangleino1/Indoor-Map-Designer/blob/main/%E6%89%AB%E7%A0%81_%E6%90%9C%E7%B4%A2%E8%81%94%E5%90%88%E4%BC%A0%E6%92%AD%E6%A0%B7%E5%BC%8F-%E6%A0%87%E5%87%86%E8%89%B2%E7%89%88.png)



# 数据集
csv 文件格式说明：`antenna_1_2_6.csv` → `antenna_天线号_坐标x_坐标y.csv`

[全量csv数据集](https://pan.quark.cn/s/be9b44dd75b6)
[全量dat原始数据集](https://pan.quark.cn/s/b2349706d0f6)

注意：`dataset` 目录中提供了少量数据集用于快速测试和演示，并非完整数据集。

# 文件说明
本项目基于 PyTorch Lightning 构建，主要文件包括：
```
main.py                     # 训练、测试统一入口脚本
base_model.py               # 所有模型共用的回归训练/评估/可视化基类
csi_dataset.py              # Dataset 和 DataModule，输出 float32 [x,y] 坐标目标
cnn_net_model.py            # 纯 CNN 回归模型
cnn_lstm_net_model.py       # CNN + LSTM 回归模型
cnn_transformer_model.py    # CNN + Transformer 回归模型
util.py                     # 数据预处理工具 (归一化、中值滤波)
data_process.py             # (可选) 原始 .dat → .csv 转换脚本
heatmappic.py               # (可选) CSI 数据热力图，用于数据质量检查
visualize_locations.py      # 可视化数据集中位置点的空间分布
USAGE_GUIDE.md              # 详细使用指南
readme.md                   # 本文件
csi.pdf                     # 参考论文
```

# Python 实现说明

## 数据集处理 (`csi_dataset.py`)
- 采用 PyTorch 的 `Dataset` 和 `DataLoader` 机制高效处理 CSI 数据。
- 输入数据为 **6 通道**：3 个天线 × (幅度 + 相位)，形状 `(6, time_step, num_subcarriers)`。
- 每个样本的目标为 **float32 张量 `[x, y]`**，即对应位置的物理坐标（网格单元）。
- 使用滑动窗口（`time_step`、`stride`）从连续 CSI 测量序列中生成训练样本。
- 数据集自动扫描目录，发现所有唯一位置点。

## 网络模型
本项目提供三种基于 PyTorch Lightning 的深度学习回归模型，均继承自 `CSIBaseModel`：

### 1. CNN 模型 (`cnn_net_model.py`)
- **架构**：3 层 Conv2d（保持空间尺寸）→ Flatten → FC(1024) → FC(512) → FC(2)
- **特点**：结构简单，训练速度快，适合快速基线实验。

### 2. CNN + LSTM 模型 (`cnn_lstm_net_model.py`)（**推荐**）
- **架构**：3 层 Conv2d → Reshape → LSTM(hidden=100) → 取最后时间步 → FC(2)
- **特点**：同时捕获空间特征（CNN）和时序依赖（LSTM），通常取得最佳精度。

### 3. CNN + Transformer 模型 (`cnn_transformer_model.py`)
- **架构**：2 层 Conv2d + MaxPool → Reshape 为序列 → TransformerEncoder → 取首 token → FC(2)
- **特点**：自注意力机制捕获长程依赖，复杂场景下可能有优势；对超参数较敏感。

所有模型均：
- 输出层为 `nn.Linear(feature_dim, 2)`，直接预测连续坐标 `(x, y)`
- 支持三种回归损失（`--reg_loss`）：`smooth_l1`（默认）、`mse`、`mae`
- 训练监控指标：`train_loss`、`train_dist`、`val_loss`、`val_dist`
- 测试结果：平均/中位距离误差、N 单元命中率、误差 CDF 图、散点图

## 训练与评估 (`main.py`)
- 基于 PyTorch Lightning，自动处理 GPU 加速、日志、检查点。
- 集成早停（`EarlyStopping`）和学习率自动衰减（`ReduceLROnPlateau`）。
- 训练结束后自动在测试集上评估并保存可视化结果。

## 回归实现关键点

1. **输出层**：预测连续 (x, y) 坐标
   ```python
   self.fc_reg = nn.Linear(feature_dim, 2)
   ```

2. **损失函数**（可选，通过 `--reg_loss` 指定）
   ```python
   loss = F.smooth_l1_loss(pred, target)  # target: float32 [x, y]
   ```

3. **距离误差**（训练时实时监控）
   ```python
   dist = torch.sqrt(((pred - target) ** 2).sum(dim=1)).mean()
   ```

4. **测试指标**
   - 平均欧氏距离误差（Mean Distance Error）
   - 中位距离误差（Median Distance Error）
   - Within-N 命中率：预测点与真实点距离 ≤ N 网格单元的样本比例
   - 误差 CDF 图 + 真实 vs 预测散点图（保存至 `results/<model>_results/`）

## 详细使用说明

有关环境配置、数据准备、模型训练、参数调整和结果分析的完整步骤，请参阅 **[详细使用指南 (USAGE_GUIDE.md)](./USAGE_GUIDE.md)**。

# Email
zhangleilikejay@gmail.com

# 技术细节

## 环境要求
- Python 3.8+
- PyTorch ≥ 1.12（推荐支持 CUDA 版本）
- PyTorch Lightning ≥ 1.6
- 详细依赖请参见 `USAGE_GUIDE.md`。

## 数据格式
- CSV 文件命名：`antenna_<天线号>_<x坐标>_<y坐标>.csv`
- 每行为一个时间点的 CSI 测量，列格式：`amplitude_0 … amplitude_29, phase_0 … phase_29`
- 每个位置需有 `antenna_1`, `antenna_2`, `antenna_3` 三个文件

## 指纹地图

![map](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning/blob/main/images/wechat_2025-08-21_093130_935.png?raw=true)

# 报错注意安装
```powershell
# 参照 USAGE_GUIDE.md 中的详细环境配置步骤
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install scikit-learn matplotlib seaborn pandas tensorboard
```

# 可能的未来优化方向
- 多传感器融合，结合其他定位数据源
- 在线学习模块，使系统适应环境变化
- 少样本学习，减少每个位置所需的训练数据量
- 联合分类 + 回归多任务学习

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
