import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes=0):
        super().__init__()
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
        self.num_classes = num_classes
        self.test_preds = []
        self.test_targets = []
        
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # 计算卷积层后的输出尺寸，以便为全连接层设置正确的输入维度
        # 假设卷积层不会改变空间尺寸（由于padding=2），则维度仍然是30x30
        conv_output_size = 18 * 30 * 30  # 18个过滤器，每个过滤器30x30的输出
        
        # 选择合适的神经元数量
        self.fc1 = nn.Linear(conv_output_size, 9000)
        self.bn4 = nn.BatchNorm1d(num_features=9000)
        self.fc2 = nn.Linear(9000, 900)
        self.bn5 = nn.BatchNorm1d(num_features=900)
        self.fc3 = nn.Linear(900, num_classes)  # 输出层，预测类别数量的神经元

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
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
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        
        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, cmap='Blues')
        plt.colorbar()
        plt.title(f'CNN Classification Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        
        # 添加数字标签到混淆矩阵
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]), 
                         ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
        
        plt.savefig(os.getcwd() + '/cnn_confusion_matrix.png')
        
        # 清空测试预测和目标列表，为下一次测试准备
        self.test_preds = []
        self.test_targets = []
        print('on_test_epoch_end completed')