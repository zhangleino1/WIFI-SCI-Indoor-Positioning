# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import CSIBaseModel


class CNN_Net(CSIBaseModel):
    """
    Pure CNN model for WiFi CSI indoor positioning (regression).

    Architecture
    ------------
    Input  (batch, 6, time_step, num_subcarriers)
      → Conv2d(6→18, k=5, p=2) + BN + ReLU   ×3  [spatial dims preserved]
      → Flatten
      → FC(18·T·S → 1024) + BN + ReLU + Dropout(0.5)
      → FC(1024 → 512)    + BN + ReLU + Dropout(0.3)
      → fc_reg: FC(512 → 2)   [predicts (x, y)]
    """

    model_name = 'cnn'

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

        feat = 18 * time_step * num_subcarriers   # dims preserved by padding

        self.fc1      = nn.Linear(feat, 1024)
        self.bn4      = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(1024, 512)
        self.bn5      = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self._init_regression_head(feature_dim=512)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.bn4(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn5(self.fc2(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_reg(self._extract_features(x))
