# CSI 数据集格式说明

## 概述

- **文件格式**: MATLAB v7.3 (基于 HDF5 的 `.mat` 文件)
- **硬件平台**: USRP X310
- **载波频率**: 2.45 GHz (WiFi 2.4G 频段)
- **采样率**: 1 MHz
- **FFT 长度**: 128
- **接收天线**: 2 (通道 1 & 通道 2)
- **子载波**: 100 (有效子载波索引: -50 到 +50)
- **每个文件数据包数量**: 1000
- **文件总数**: 24 (覆盖 4x6 网格)
- **采集日期**: 2026-05-20

## 文件命名规范

```
CSI_RX_x_p{X}d{DEC}_y_p{Y}d{DEC}_z_p{Z}d{DEC}_{DATE}_{TIME}.mat
```

| 字段 | 含义 | 示例 |
|-------|---------|---------|
| `CSI_RX` | 固定前缀，CSI 接收器数据 | - |
| `x_p3d000` | X 坐标 = +3.000m，`p`=正数，`d`=小数点 | x=3.0 |
| `y_p2d000` | Y 坐标 = +2.000m | y=2.0 |
| `z_p0d000` | Z 坐标 = +0.000m (全部在地面层) | z=0.0 |
| `20260520` | 日期: 2026-05-20 | - |
| `155037` | 时间: 15:50:37 | - |

**坐标网格**: x 在 {0, 1, 2, 3} 米，y 在 {0, 1, 2, 3, 4, 5} 米，z = 0

## 数据结构 (HDF5 键名)

### 核心 CSI 数据 (形状: `(2, 100, 1000)` = `(接收天线数, 子载波数, 数据包数)`)

| 键名 | 描述 |
|-----|-------------|
| `csiRaw` | 原始复数 CSI (数据类型: complex128) |
| `csiAmplitudeRaw` | 原始幅度，范围 [0.00007, 0.136] |
| `csiAmplitudeMedian` | 中值滤波后的幅度 |
| `csiAmplitudeOutlierRemoved` | 移除异常值后的幅度 |
| `csiAmplitudeFiltered` | 最终平滑幅度 (推荐用于训练) |
| `csiPhaseRaw` | 原始相位 [-pi, pi] |
| `csiPhaseUnwrap` | 解卷绕后的相位 |
| `csiPhaseCalibrated` | 校准后的相位 (移除线性分量，推荐使用) |
| `csiPhaseCalibratedWrapped` | 重新缠绕至 [-pi, pi] 的校准后相位 |
| `csiPhaseLinearFit` | 用于相位校准的线性拟合 |

### 特征数据

| 键名 | 形状 | 描述 |
|-----|-------|-------------|
| `featureAmpMean` | `(2, 100)` | 每个天线每个子载波的平均幅度 |
| `featurePhaseMean` | `(2, 100)` | 每个天线每个子载波的平均相位 |
| `featureVector` | `(1, 400)` | 拼接特征: [ant0_amp(100), ant1_amp(100), ant0_phase(100), ant1_phase(100)] |

### 元数据

| 键名 | 形状 | 描述 |
|-----|-------|-------------|
| `coord/x`, `coord/y`, `coord/z` | `(1,1)` | 位置坐标 (米) |
| `coord/tag` | 字符串 | 位置标签，例如 `x_p0d000_y_p0d000_z_p0d000` |
| `activeSubcarrierIndex` | `(1, 100)` | 子载波索引 [-50, +50] |
| `packetStartIndex` | `(1, 1000)` | 每个数据包起始的样本索引 |
| `phaseSlope` | `(2, 1000)` | 每个天线每个数据包的线性相位斜率 |
| `phaseIntercept` | `(2, 1000)` | 每个天线每个数据包的线性相位截距 |

### 配置 (`cfg/`)

| 键名 | 值 | 描述 |
|-----|-------|-------------|
| `nRxAnt` | 2 | 接收天线数量 |
| `nPackets` | 1000 | 每个位置收集的数据包数量 |
| `nSubcarriers` | 100 | 有效 OFDM 子载波数量 |
| `centerFrequency` | 2.45e9 | 中心频率 (Hz) |
| `sampleRate` | 1e6 | 采样率 (Hz) |
| `fftLen` | 128 | FFT 长度 |
| `rxChannel` | [1, 2] | 接收通道索引 |
| `rxGain` | 25 | 接收器增益 (dB) |
| `platform` | X310 | USRP 平台型号 |

## 推荐用于模型训练的数据

- **幅度**: 使用 `csiAmplitudeFiltered` (平滑且移除异常值)
- **相位**: 使用 `csiPhaseCalibrated` (已移除线性分量) 或 `csiPhaseCalibratedWrapped`
- **输入张量形状**: `(batch, 4, time_step, 100)`，其中 4 = 2 天线 x 2 特征 (幅度 + 相位)
- **目标值**: 来自 `coord/x` 和 `coord/y` 的 `(x, y)` 坐标
