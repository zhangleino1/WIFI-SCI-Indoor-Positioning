import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix

class CNN_Transformer_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes, 
                 time_step, num_subcarriers, d_model_override=None, # d_model will be calculated, override for testing
                 nhead=4, num_encoder_layers=3, dim_feedforward=256):
        super().__init__()
        self.save_hyperparameters()

        self.test_preds = []
        self.test_targets = []
        self.printed_input_shape = False


        # CNN Backbone
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves dimensions: H/2, W/2

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves dimensions again: H/4, W/4

        # Calculate the output dimensions of CNN
        self.cnn_out_height = self.hparams.time_step // 4
        self.cnn_out_width = self.hparams.num_subcarriers // 4
        
        # d_model for transformer is features per token/patch
        # Each token corresponds to a "pixel" from the CNN's output feature map height dimension
        # The features for that token are the channels (32) times the width of the feature map (cnn_out_width)
        if d_model_override is not None:
            self.d_model = d_model_override # Allow override for specific testing/tuning
        else:
            self.d_model = 32 * self.cnn_out_width 

        if self.d_model % self.hparams.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.hparams.nhead})")

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.hparams.nhead, 
            dim_feedforward=self.hparams.dim_feedforward, 
            batch_first=True, # Expects (batch, seq_len, features)
            dropout=0.1 # Standard dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.hparams.num_encoder_layers)

        # Classification Head
        # We take the output corresponding to the first token from the sequence
        self.fc = nn.Linear(self.d_model, self.hparams.num_classes)

    def forward(self, x):
        if not self.printed_input_shape and hasattr(x, 'ndim') and x.ndim == 4:
            print(f"CNN_Transformer_Net input shape: {x.shape}") # (batch, 6, time_step, num_subcarriers)
            self.printed_input_shape = True

        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) 
        # Shape: (batch, 16, time_step/2, num_subcarriers/2)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) 
        # Shape: (batch, 32, time_step/4, num_subcarriers/4) -> (batch, 32, cnn_out_height, cnn_out_width)
        
        # Prepare for Transformer: (batch, seq_len, features)
        # Here, seq_len is cnn_out_height, and features per item in seq is 32 * cnn_out_width
        x = x.permute(0, 2, 1, 3).contiguous() 
        # Shape: (batch, cnn_out_height, 32, cnn_out_width)
        x = x.reshape(x.size(0), self.cnn_out_height, -1) 
        # Shape: (batch, cnn_out_height, 32 * cnn_out_width) 
        # This is (batch, seq_len, d_model) where seq_len = cnn_out_height
        
        # Optional: Add positional encoding here if implemented
        # x = self.pos_encoder(x) 

        x = self.transformer_encoder(x) 
        # Output shape: (batch, cnn_out_height, d_model)
        
        x = x[:, 0, :]  # Take the output of the first token/patch in the sequence
        # Shape: (batch, d_model)
        
        x = self.fc(x)
        # Shape: (batch, num_classes)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.hparams.lr_factor, 
            patience=self.hparams.lr_patience, 
            eps=self.hparams.lr_eps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
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
        return loss

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
            print("Warning: test_targets or test_preds is empty in on_test_epoch_end for CNN_Transformer_Net. Skipping confusion matrix.")
            self.test_preds = []
            self.test_targets = []
            return

        accuracy = accuracy_score(self.test_targets, self.test_preds)
        conf_matrix = confusion_matrix(self.test_targets, self.test_preds)

        print(f"CNN_Transformer_Net Test Accuracy: {accuracy:.4f}")
        print("CNN_Transformer_Net Confusion Matrix:")
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
            plt.title(f'CNN_Transformer_Net Classification Confusion Matrix\nAccuracy: {accuracy:.4f}')
            
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
            log_dir = os.path.join(os.getcwd(), "results", "cnn_transformer_results") # Specific subdir
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, 'cnn_transformer_net_confusion_matrix.png')
            try:
                plt.savefig(save_path, dpi=300)
                print(f"CNN_Transformer_Net Confusion matrix saved to {save_path}")
            except Exception as e:
                print(f"Error saving CNN_Transformer_Net confusion matrix: {e}")
            plt.close()
        else:
            print("Warning: class_to_location mapping not available for CNN_Transformer_Net. Skipping confusion matrix labels and saving.")

        self.test_preds = []
        self.test_targets = []
        print('CNN_Transformer_Net on_test_epoch_end completed')
