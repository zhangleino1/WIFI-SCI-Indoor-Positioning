import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps):
        super().__init__()
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
        self.test_losses = []  # 初始化空列表以收集测试损失
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # LSTM层
        self.lstm = nn.LSTM(input_size=18*30, hidden_size=100, batch_first=True)  # 每行作为一个时间步

        # 输出层
        self.fc = nn.Linear(100, 2)  # 输出二维坐标

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 重塑输出以匹配LSTM输入 (batch_size, seq_len, features)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, height, channels, width)
        x = x.view(x.size(0), x.size(1), -1)  # (batch, 30, 18*30)

        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 经过一个全连接层输出最终的二维坐标
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience=self.lr_patience, eps=self.lr_eps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = F.mse_loss(outputs, targets)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = F.mse_loss(outputs, targets)
        self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = F.mse_loss(outputs, targets)
        self.log('test_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_losses.append(loss.item())  # 将损失加入列表
        return loss

     # 每个epoch结束后执行,一个test_data_loader执行一次
    def on_test_epoch_end(self):
        # 将损失列表转换为NumPy数组并四舍五入到指定的小数位数
        losses = np.round(np.array(self.test_losses), decimals=2)  # 例如四舍五入到小数点后两位
        unique_losses, counts = np.unique(losses, return_counts=True)
        # 计算每个唯一损失值的累积分布百分比
        cumulative_distribution = np.cumsum(counts) / np.sum(counts)

        # 绘制CDF图，确保不包含平直线
        plt.figure(figsize=(8, 6))
        plt.step(unique_losses, cumulative_distribution,linestyle='-', linewidth=2)  # 使用step绘图，避免线性插值
        plt.xlabel('Distance Error (meter)')
        plt.ylabel('CNN_LSTM CDF')
        plt.grid(True)
        plt.savefig(os.getcwd() + '/cnn_lstm_cdf.png')  # 保存图像
        plt.show()

        # 清空测试损失列表，为下一次测试准备
        self.test_losses = []
        print('on_test_epoch_end')