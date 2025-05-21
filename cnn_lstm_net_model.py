import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes=0):
        super().__init__()
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
        self.num_classes = num_classes
        self.test_preds = []
        self.test_targets = []
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # LSTM层
        self.lstm = nn.LSTM(input_size=18*30, hidden_size=100, batch_first=True)  # 每行作为一个时间步

        # 输出层 - 修改为输出到类别数量
        self.fc = nn.Linear(100, num_classes)

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

        # 经过一个全连接层输出类别预测
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # When monitoring accuracy, we want to maximize it
            factor=self.lr_factor, 
            patience=self.lr_patience, 
            eps=self.lr_eps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',  # Monitor validation accuracy
            },
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        # self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # 保存预测和目标，用于计算混淆矩阵
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(targets.cpu().numpy())
        
        return loss

    # 每个epoch结束后执行,一个test_data_loader执行一次
    def on_test_epoch_end(self):
        # 计算准确率和混淆矩阵
        accuracy = accuracy_score(self.test_targets, self.test_preds)
        conf_matrix = confusion_matrix(self.test_targets, self.test_preds)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # 获取数据集的类别到位置的映射
        # 假设我们可以通过trainer.datamodule访问数据集
        class_to_location = self.trainer.datamodule.dataset.class_to_location
        
        # 可视化混淆矩阵
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, cmap='Blues')
        plt.colorbar()
        plt.title(f'CNN_LSTM Classification Confusion Matrix\nAccuracy: {accuracy:.4f}')
        
        # 生成坐标标签
        classes = sorted(list(class_to_location.keys()))
        location_labels = [f"({class_to_location[c][0]},{class_to_location[c][1]})" for c in classes]
        
        # 设置轴标签
        plt.xlabel('Predicted Location')
        plt.ylabel('True Location')
        
        # 设置刻度位置和标签
        tick_positions = np.arange(len(classes))
        plt.xticks(tick_positions, location_labels, rotation=90, fontsize=8)
        plt.yticks(tick_positions, location_labels, fontsize=8)
        
        # 添加数字标签到混淆矩阵
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]), 
                         ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
        
        plt.tight_layout()
        plt.savefig(os.getcwd() + '/cnn_lstm_confusion_matrix.png', dpi=300)
        
        # 清空测试预测和目标列表，为下一次测试准备
        self.test_preds = []
        self.test_targets = []
        print('on_test_epoch_end completed')