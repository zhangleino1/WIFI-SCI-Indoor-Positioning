import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, dropout_rate=0.5):
        super().__init__()
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
        self.test_losses = []

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        self.lstm1 = nn.LSTM(input_size=30*30, hidden_size=256, batch_first=True)
        self.bn_lstm1 = nn.BatchNorm1d(num_features=256)  # 批量归一化层
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout层

        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.bn_lstm2 = nn.BatchNorm1d(num_features=128)  # 另一个批量归一化层

        self.fc1 = nn.Linear(128, 5000)
        self.bn4 = nn.BatchNorm1d(num_features=5000)
        self.fc2 = nn.Linear(5000, 500)
        self.bn5 = nn.BatchNorm1d(num_features=500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.lstm1(x)
        x = self.bn_lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.bn_lstm2(x)

        x = x[:, -1, :]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
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
        # 将损失列表转换为NumPy数组，计算CDF
        losses = np.array(self.test_losses)
        unique_losses, counts = np.unique(losses, return_counts=True)
        # 计算每个唯一损失值的累积分布百分比
        cumulative_distribution = np.cumsum(counts) / np.sum(counts)

        # 绘制CDF图，确保不包含平直线
        plt.figure(figsize=(8, 6))
        plt.step(unique_losses, cumulative_distribution, where='post')  # 使用step绘图，避免线性插值
        plt.xlabel('Distnce Error (meter)')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.savefig(os.getcwd() + '/test_loss_cdf.png')  # 保存图像
        plt.show()

        # 清空测试损失列表，为下一次测试准备
        self.test_losses = []
        print('on_test_epoch_end')