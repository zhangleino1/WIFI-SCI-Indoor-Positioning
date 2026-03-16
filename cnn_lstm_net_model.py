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
    Input  (batch, 6, time_step, num_subcarriers)
      → Conv2d(6→18, k=5, p=2) + BN + ReLU   ×3  [spatial dims preserved]
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
        num_subcarriers: int = 30,
        reg_loss: str = 'smooth_l1',   # 'smooth_l1' | 'mse' | 'mae'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(6, 18, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(18)
        self.conv2 = nn.Conv2d(18, 18, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(18)
        self.conv3 = nn.Conv2d(18, 18, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm2d(18)

        self.lstm = nn.LSTM(
            input_size=18 * num_subcarriers,
            hidden_size=100,
            batch_first=True,
        )

        self._init_regression_head(feature_dim=100)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # (batch, 18, T, S) → (batch, T, 18·S)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        return x[:, -1, :]    # last time-step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_reg(self._extract_features(x))
