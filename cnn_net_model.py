import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes=0, time_step=15, num_subcarriers=30): # Added time_step and num_subcarriers
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ arguments to self.hparams
        # self.lr = lr # Not needed if save_hyperparameters() is used and accessed via self.hparams.lr
        # self.lr_factor = lr_factor
        # self.lr_patience = lr_patience
        # self.lr_eps = lr_eps
        # self.num_classes = num_classes
        # self.time_step = time_step
        # self.num_subcarriers = num_subcarriers
        
        self.test_preds = []
        self.test_targets = []
        self.printed_input_shape = False # For printing input shape once
        
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # Calculate conv_output_size dynamically based on time_step and num_subcarriers
        # Conv layers with kernel_size=5, padding=2 maintain HxW dimensions
        conv_output_size = 18 * self.hparams.time_step * self.hparams.num_subcarriers
        
        self.fc1 = nn.Linear(conv_output_size, 9000)
        self.bn4 = nn.BatchNorm1d(num_features=9000)
        self.fc2 = nn.Linear(9000, 900)
        self.bn5 = nn.BatchNorm1d(num_features=900)
        self.fc3 = nn.Linear(900, self.hparams.num_classes)

    def forward(self, x):
        if not self.printed_input_shape and x.ndim == 4:
            print(f"CNN_Net input shape: {x.shape}") # Expected (batch_size, 6, time_step, num_subcarriers)
            self.printed_input_shape = True
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # Use self.hparams
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Changed to 'min' for val_loss
            factor=self.hparams.lr_factor, 
            patience=self.hparams.lr_patience, 
            eps=self.hparams.lr_eps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Changed to 'val_loss'
            },
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        return loss # Return loss for ReduceLROnPlateau

    def test_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True) # on_step=False
        
        # 保存预测和目标，用于计算混淆矩阵
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(targets.cpu().numpy())
        
        return loss

    # 每个epoch结束后执行,一个test_data_loader执行一次
    def on_test_epoch_end(self):
        if not self.test_targets or not self.test_preds:
            print("Warning: test_targets or test_preds is empty in on_test_epoch_end for CNN_Net. Skipping confusion matrix.")
            # Ensure lists are cleared even if empty, for subsequent runs if any
            self.test_preds = []
            self.test_targets = []
            return

        accuracy = accuracy_score(self.test_targets, self.test_preds)
        conf_matrix = confusion_matrix(self.test_targets, self.test_preds)
        
        print(f"CNN_Net Test Accuracy: {accuracy:.4f}")
        print("CNN_Net Confusion Matrix:")
        print(conf_matrix)
        
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule and hasattr(datamodule, 'dataset') and \
           datamodule.dataset is not None and \
           hasattr(datamodule.dataset, 'class_to_location'):
            class_to_location = datamodule.dataset.class_to_location
            classes = sorted(list(class_to_location.keys()))
            location_labels = [f"({class_to_location[c][0]},{class_to_location[c][1]})" for c in classes]

            # Adjust figure size dynamically
            fig_width = max(12, len(classes) * 0.6)
            fig_height = max(10, len(classes) * 0.5)
            plt.figure(figsize=(fig_width, fig_height))

            plt.imshow(conf_matrix, cmap='Blues')
            plt.colorbar()
            plt.title(f'CNN_Net Classification Confusion Matrix\nAccuracy: {accuracy:.4f}')
            
            plt.xlabel('Predicted Location')
            plt.ylabel('True Location')
            
            tick_positions = np.arange(len(classes))
            plt.xticks(tick_positions, location_labels, rotation=90, fontsize=8)
            plt.yticks(tick_positions, location_labels, fontsize=8)
            
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, str(conf_matrix[i, j]), 
                             ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
            
            plt.tight_layout()
            # Ensure logs directory exists
            log_dir = os.path.join(os.getcwd(), "results", "cnn_results") # Specific subdir for CNN results
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, 'cnn_net_confusion_matrix.png')
            try:
                plt.savefig(save_path, dpi=300)
                print(f"CNN_Net Confusion matrix saved to {save_path}")
            except Exception as e:
                print(f"Error saving confusion matrix: {e}")
            plt.close() # Close figure to free memory
        else:
            print("Warning: class_to_location mapping not available for CNN_Net. Skipping confusion matrix labels and saving.")

        self.test_preds = []
        self.test_targets = []
        print('CNN_Net on_test_epoch_end completed')