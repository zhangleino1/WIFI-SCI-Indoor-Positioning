# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import CSIBaseModel


class CNN_Transformer_Net(CSIBaseModel):
    """
    CNN + Transformer model for WiFi CSI indoor positioning (regression).

    Architecture
    ------------
    Input  (batch, 6, time_step, num_subcarriers)
      → Conv2d(6→16, k=3, p=1) + BN + ReLU + MaxPool(2,2)
          → (batch, 16, T/2, S/2)
      → Conv2d(16→32, k=3, p=1) + BN + ReLU + MaxPool(2,2)
          → (batch, 32, T/4, S/4)
      → Reshape → (batch, T/4, 32·S/4)   [sequence for Transformer]
      → TransformerEncoder(d_model=32·S/4, nhead, layers, ff_dim)
      → First token → fc_reg: FC(d_model → 2)   [predicts (x, y)]

    Notes
    -----
    - With default time_step=15, num_subcarriers=30:
        after two MaxPool(2,2): height = floor(15/2/2) = 3,
                                width  = floor(30/2/2) = 7
        d_model = 32 * 7 = 224  (divisible by nhead=4 ✓)
    - d_model must be divisible by nhead; adjust nhead accordingly.
    """

    model_name = 'cnn_transformer'

    def __init__(
        self,
        lr: float,
        lr_factor: float,
        lr_patience: int,
        lr_eps: float,
        num_classes: int = 0,           # unused; kept for API compat
        time_step: int = 15,
        num_subcarriers: int = 30,
        reg_loss: str = 'smooth_l1',    # 'smooth_l1' | 'mse' | 'mae'
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- CNN backbone ----
        self.conv1 = nn.Conv2d(6,  16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two MaxPool(2,2): floor(dim / 4)
        self._cnn_h = time_step      // 4   # sequence length for Transformer
        self._cnn_w = num_subcarriers // 4   # features per token
        d_model = 32 * self._cnn_w

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
                f"Adjust num_subcarriers or nhead."
            )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self._init_regression_head(feature_dim=d_model)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 16, T/2, S/2)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 32, T/4, S/4)
        # → (B, seq_len=T/4, d_model=32·S/4)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(x.size(0), self._cnn_h, -1)
        x = self.transformer_encoder(x)
        return x[:, 0, :]    # first token as global representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_reg(self._extract_features(x))
