import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes=0, time_step=15, num_subcarriers=30): # Added time_step, num_subcarriers
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ arguments to self.hparams
        
        self.test_preds = []
        self.test_targets = []
        self.printed_input_shape = False # For printing input shape once
        
        # 卷积层 (Conv layers are same as CNN_Net, preserving HxW)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # LSTM层
        # Input to LSTM: (batch, seq_len, features)
        # seq_len is time_step (self.hparams.time_step)
        # features is 18 * num_subcarriers (18 channels from CNN, each with num_subcarriers features)
        lstm_input_features = 18 * self.hparams.num_subcarriers
        self.lstm = nn.LSTM(input_size=lstm_input_features, hidden_size=100, batch_first=True)

        # 输出层
        self.fc = nn.Linear(100, self.hparams.num_classes) # LSTM hidden_size to num_classes

    def forward(self, x):
        if not self.printed_input_shape and hasattr(x, 'ndim') and x.ndim == 4:
            print(f"CNN_LSTM_Net input shape: {x.shape}") # Expected (batch, 6, time_step, num_subcarriers)
            self.printed_input_shape = True

        # CNN part
        x = F.relu(self.bn1(self.conv1(x))) # Output: (batch, 18, time_step, num_subcarriers)
        x = F.relu(self.bn2(self.conv2(x))) # Output: (batch, 18, time_step, num_subcarriers)
        x = F.relu(self.bn3(self.conv3(x))) # Output: (batch, 18, time_step, num_subcarriers)
        
        # Reshape for LSTM
        # (batch, channels_cnn, time_step, num_subcarriers) -> (batch, time_step, channels_cnn * num_subcarriers)
        # Here, time_step is the sequence length for LSTM
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time_step, 18, num_subcarriers)
        # x.size(1) is time_step, x.size(2) is 18 (channels_cnn), x.size(3) is num_subcarriers
        x = x.view(x.size(0), x.size(1), -1)  # (batch, time_step, 18 * num_subcarriers)

        # LSTM processing
        x, _ = self.lstm(x) # Output: (batch, time_step, lstm_hidden_size)
        x = x[:, -1, :]  # Take the output from the last time step (batch, lstm_hidden_size)

        # Fully connected layer
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Monitor val_loss
            factor=self.hparams.lr_factor, 
            patience=self.hparams.lr_patience, 
            eps=self.hparams.lr_eps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor val_loss
            },
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss # Crucial: return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(targets.cpu().numpy())
        
        return loss

    def on_test_epoch_end(self):
        if not self.test_targets or not self.test_preds:
            print("Warning: test_targets or test_preds is empty in on_test_epoch_end for CNN_LSTM_Net. Skipping confusion matrix.")
            self.test_preds = []
            self.test_targets = []
            return

        accuracy = accuracy_score(self.test_targets, self.test_preds)
        conf_matrix = confusion_matrix(self.test_targets, self.test_preds)
        
        print(f"CNN_LSTM_Net Test Accuracy: {accuracy:.4f}")
        print("CNN_LSTM_Net Confusion Matrix:")
        print(conf_matrix)
        
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule and hasattr(datamodule, 'dataset') and \
           datamodule.dataset is not None and \
           hasattr(datamodule.dataset, 'class_to_location'):
            class_to_location = datamodule.dataset.class_to_location
            classes = sorted(list(class_to_location.keys()))
            location_labels = [f"({class_to_location[c][0]},{class_to_location[c][1]})" for c in classes]

            fig_width = max(12, len(classes) * 0.6)
            fig_height = max(10, len(classes) * 0.5)
            plt.figure(figsize=(fig_width, fig_height))
            
            plt.imshow(conf_matrix, cmap='Blues')
            plt.colorbar()
            plt.title(f'CNN_LSTM_Net Classification Confusion Matrix\nAccuracy: {accuracy:.4f}')
            
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
            log_dir = os.path.join(os.getcwd(), "results", "cnn_lstm_results") # Specific subdir
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, 'cnn_lstm_net_confusion_matrix.png')
            try:
                plt.savefig(save_path, dpi=300)
                print(f"CNN_LSTM_Net Confusion matrix saved to {save_path}")
            except Exception as e:
                print(f"Error saving CNN_LSTM_Net confusion matrix: {e}")
            plt.close()
        else:
            print("Warning: class_to_location mapping not available for CNN_LSTM_Net. Skipping confusion matrix labels and saving.")

        self.test_preds = []
        self.test_targets = []
        print('CNN_LSTM_Net on_test_epoch_end completed')