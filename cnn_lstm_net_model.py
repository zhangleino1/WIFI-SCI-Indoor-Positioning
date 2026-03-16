# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix


class CNN_LSTM_Net(pl.LightningModule):
    def __init__(self, lr, lr_factor, lr_patience, lr_eps, num_classes=0,
                 time_step=15, num_subcarriers=30, task='classification'):
        super().__init__()
        self.save_hyperparameters()

        self.test_preds = []
        self.test_targets = []
        self.printed_input_shape = False

        # CNN layers (preserve HxW with padding=2)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=18)

        # LSTM: (batch, time_step, 18 * num_subcarriers)
        lstm_input_features = 18 * self.hparams.num_subcarriers
        self.lstm = nn.LSTM(input_size=lstm_input_features, hidden_size=100, batch_first=True)

        # Output: 2 neurons for regression (x,y), num_classes for classification
        out_features = 2 if self.hparams.task == 'regression' else self.hparams.num_classes
        self.fc = nn.Linear(100, out_features)

    def forward(self, x):
        if not self.printed_input_shape and hasattr(x, 'ndim') and x.ndim == 4:
            print(f"CNN_LSTM_Net input shape: {x.shape}")
            self.printed_input_shape = True

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # (batch, 18, time_step, num_subcarriers) -> (batch, time_step, 18 * num_subcarriers)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last time step
        x = self.fc(x)
        return x

    def _compute_loss(self, output, targets):
        if self.hparams.task == 'regression':
            return F.smooth_l1_loss(output, targets)
        else:
            return F.cross_entropy(output, targets)

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
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch
        output = self(data)
        loss = self._compute_loss(output, targets)

        if self.hparams.task == 'regression':
            dist = torch.sqrt(((output - targets) ** 2).sum(dim=1)).mean()
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_dist', dist, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            preds = torch.argmax(output, dim=1)
            acc = (preds == targets).float().mean()
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        output = self(data)
        loss = self._compute_loss(output, targets)

        if self.hparams.task == 'regression':
            dist = torch.sqrt(((output - targets) ** 2).sum(dim=1)).mean()
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_dist', dist, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            preds = torch.argmax(output, dim=1)
            acc = (preds == targets).float().mean()
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        output = self(data)
        loss = self._compute_loss(output, targets)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.hparams.task == 'regression':
            self.test_preds.extend(output.cpu().detach().numpy())
            self.test_targets.extend(targets.cpu().numpy())
        else:
            preds = torch.argmax(output, dim=1)
            acc = (preds == targets).float().mean()
            self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.test_preds.extend(preds.cpu().numpy())
            self.test_targets.extend(targets.cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if not self.test_targets or not self.test_preds:
            self.test_preds = []
            self.test_targets = []
            return

        log_dir = os.path.join(os.getcwd(), "results", "cnn_lstm_results")
        os.makedirs(log_dir, exist_ok=True)

        if self.hparams.task == 'regression':
            preds = np.array(self.test_preds)    # shape (N, 2)
            targets = np.array(self.test_targets)  # shape (N, 2)
            distances = np.sqrt(((preds - targets) ** 2).sum(axis=1))
            mean_dist = distances.mean()
            median_dist = np.median(distances)
            within_1 = (distances <= 1.0).mean() * 100
            within_2 = (distances <= 2.0).mean() * 100
            within_3 = (distances <= 3.0).mean() * 100

            print(f"CNN_LSTM_Net Regression Results:")
            print(f"  Mean Distance Error:   {mean_dist:.4f} units")
            print(f"  Median Distance Error: {median_dist:.4f} units")
            print(f"  Within 1 unit:  {within_1:.1f}%")
            print(f"  Within 2 units: {within_2:.1f}%")
            print(f"  Within 3 units: {within_3:.1f}%")

            sorted_dist = np.sort(distances)
            cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
            plt.figure(figsize=(8, 5))
            plt.plot(sorted_dist, cdf)
            plt.xlabel('Distance Error (grid units)')
            plt.ylabel('CDF')
            plt.title(f'CNN_LSTM_Net Regression - CDF of Distance Error\nMean: {mean_dist:.3f}, Median: {median_dist:.3f}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'cnn_lstm_net_regression_cdf.png'), dpi=300)
            plt.close()

            plt.figure(figsize=(8, 8))
            plt.scatter(targets[:, 0], targets[:, 1], c='blue', alpha=0.3, s=10, label='True')
            plt.scatter(preds[:, 0], preds[:, 1], c='red', alpha=0.3, s=10, label='Predicted')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('CNN_LSTM_Net Regression: True vs Predicted Locations')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'cnn_lstm_net_regression_scatter.png'), dpi=300)
            plt.close()
            print(f"CNN_LSTM_Net regression plots saved to {log_dir}")
        else:
            accuracy = accuracy_score(self.test_targets, self.test_preds)
            conf_matrix = confusion_matrix(self.test_targets, self.test_preds)

            print(f"CNN_LSTM_Net Test Accuracy: {accuracy:.4f}")
            print("CNN_LSTM_Net Confusion Matrix:")
            print(conf_matrix)

            datamodule = getattr(self.trainer, 'datamodule', None)
            if datamodule and hasattr(datamodule, 'dataset') and \
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
                                 ha="center", va="center",
                                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
                plt.tight_layout()
                save_path = os.path.join(log_dir, 'cnn_lstm_net_confusion_matrix.png')
                try:
                    plt.savefig(save_path, dpi=300)
                    print(f"CNN_LSTM_Net Confusion matrix saved to {save_path}")
                except Exception as e:
                    print(f"Error saving CNN_LSTM_Net confusion matrix: {e}")
                plt.close()
            else:
                print("Warning: class_to_location mapping not available. Skipping confusion matrix.")

        self.test_preds = []
        self.test_targets = []
        print('CNN_LSTM_Net on_test_epoch_end completed')
