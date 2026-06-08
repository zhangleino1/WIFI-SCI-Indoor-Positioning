# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import CSIBaseModel


class CNN_LSTM_Net(CSIBaseModel):
    """
    CNN + LSTM model for WiFi CSI indoor positioning (regression).

    Architecture
    ------------
    Input  (batch, in_channels, time_step, num_subcarriers)
      → Conv2d(in_channels→18, k=5, p=2) + BN + ReLU   ×3  [spatial dims preserved]
      → Reshape → (batch, time_step, 18 · num_subcarriers)
      → LSTM(hidden=100, batch_first=True)  → last time-step
      → fc_reg: FC(100 → 2)   [predicts (x, y)]
    """

    model_name = 'cnn_lstm'

    def __init__(
        self,
        lr: float,
        lr_factor: float,
        lr_patience: int,
        lr_eps: float,
        time_step: int = 15,
        num_subcarriers: int = 100,
        in_channels: int = 4,
        reg_loss: str = 'smooth_l1',   # 'smooth_l1' | 'mse' | 'mae'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(in_channels, 18, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(18)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Downsamples subcarriers only

        self.conv2 = nn.Conv2d(18, 18, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(18)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Downsamples subcarriers only

        self.conv3 = nn.Conv2d(18, 18, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm2d(18)

        # Subcarriers are downsampled by a factor of 4
        sub_dim = num_subcarriers // 4
        self.lstm = nn.LSTM(
            input_size=18 * sub_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )

        # Bidirectional output size = 2 * 256 = 512. Concatenating mean and max pool gives 1024.
        self._init_regression_head(feature_dim=1024)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        # (batch, 18, T, S_pooled) → (batch, T, 18·S_pooled)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        # Global Average Pooling and Global Max Pooling over time dimension
        x_avg = x.mean(dim=1)
        x_max, _ = x.max(dim=1)
        return torch.cat([x_avg, x_max], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_reg(self._extract_features(x))
