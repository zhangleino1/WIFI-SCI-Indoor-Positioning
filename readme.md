

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

# 报错 注意安装
```
conda install lightning -c conda-forge
pip install -U 'tensorboardX'
pip install -U 'tensorboard'

# csv文件格式说明
antenna_1_2_6.csv  antenna_天线号_坐标x_坐标y.csv

#  wifi sci 热力图
![0_0_ant0_amplitude_filtered](/)