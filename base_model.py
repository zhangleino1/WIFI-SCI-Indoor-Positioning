# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
"""
CSIBaseModel — shared regression training / evaluation / visualization logic
for all three architectures (CNN, CNN-LSTM, CNN-Transformer).

Each concrete model only needs to:
  1. Inherit from CSIBaseModel
  2. Build backbone layers in __init__, then call _init_regression_head(feature_dim)
  3. Implement _extract_features(x) -> Tensor(batch, feature_dim)
  4. Implement forward(x) -> Tensor(batch, 2)  [predicts (x, y) coordinates]

Supported regression losses (--reg_loss)
-----------------------------------------
smooth_l1  Huber loss — robust to outliers, default choice
mse        Mean Squared Error — penalises large errors more heavily
mae        Mean Absolute Error (L1) — most robust to outliers
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class CSIBaseModel(pl.LightningModule):
    """Base class — do NOT instantiate directly."""

    # Subclasses set this for result directory naming
    model_name: str = 'base'

    # ------------------------------------------------------------------
    # Initialisation helper
    # ------------------------------------------------------------------

    def _init_regression_head(self, feature_dim: int):
        """
        Call at the end of each subclass __init__ after backbone layers.
        Creates the regression output head and test accumulators.
        """
        self.fc_reg = nn.Linear(feature_dim, 2)   # outputs (x, y)

        self.test_reg_preds:   list = []
        self.test_reg_targets: list = []

    # ------------------------------------------------------------------
    # Loss helper
    # ------------------------------------------------------------------

    def _reg_loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = self.hparams.reg_loss
        if loss_type == 'mse':
            return F.mse_loss(pred, target)
        if loss_type == 'mae':
            return F.l1_loss(pred, target)
        return F.smooth_l1_loss(pred, target)   # default: smooth_l1

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str):
        data, targets = batch          # targets: float32 (batch, 2) = [x, y]
        preds = self(data)             # (batch, 2)
        loss  = self._reg_loss_fn(preds, targets)

        with torch.no_grad():
            dist = torch.sqrt(((preds - targets) ** 2).sum(dim=1)).mean()

        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True,  logger=True, on_step=False)
        self.log(f'{stage}_dist', dist, on_epoch=True, prog_bar=True,  logger=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            eps=self.hparams.lr_eps,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    # ------------------------------------------------------------------
    # Test step — accumulate predictions
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        data, targets = batch
        preds = self(data)
        loss  = self._step(batch, 'test')

        self.test_reg_preds.extend(preds.detach().cpu().numpy())
        self.test_reg_targets.extend(targets.cpu().numpy())
        return loss

    # ------------------------------------------------------------------
    # Test epoch end — metrics + plots
    # ------------------------------------------------------------------

    def on_test_epoch_end(self):
        if not self.test_reg_preds:
            return

        log_dir = os.path.join(os.getcwd(), 'results', f'{self.model_name}_results')
        os.makedirs(log_dir, exist_ok=True)
        self._report_regression(log_dir)

        self.test_reg_preds   = []
        self.test_reg_targets = []
        print(f'{self.__class__.__name__} on_test_epoch_end completed')

    # ------------------------------------------------------------------
    # Regression reporting
    # ------------------------------------------------------------------

    def _report_regression(self, log_dir: str):
        preds   = np.array(self.test_reg_preds)    # (N, 2)
        targets = np.array(self.test_reg_targets)  # (N, 2)
        dists   = np.sqrt(((preds - targets) ** 2).sum(axis=1))

        mean_d   = dists.mean()
        median_d = np.median(dists)
        w1 = (dists <= 1.0).mean() * 100
        w2 = (dists <= 2.0).mean() * 100
        w3 = (dists <= 3.0).mean() * 100

        print(f'\n{self.__class__.__name__} Regression Results:')
        print(f'  Mean Distance Error   : {mean_d:.4f} grid units')
        print(f'  Median Distance Error : {median_d:.4f} grid units')
        print(f'  Within 1 unit  : {w1:.1f}%')
        print(f'  Within 2 units : {w2:.1f}%')
        print(f'  Within 3 units : {w3:.1f}%')

        # CDF plot
        sd  = np.sort(dists)
        cdf = np.arange(1, len(sd) + 1) / len(sd)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sd, cdf, linewidth=2)
        ax.axvline(mean_d,   color='r', linestyle='--', label=f'Mean {mean_d:.2f}')
        ax.axvline(median_d, color='g', linestyle='--', label=f'Median {median_d:.2f}')
        ax.set_xlabel('Distance Error (grid units)')
        ax.set_ylabel('CDF')
        ax.set_title(f'{self.__class__.__name__} — CDF of Position Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, f'{self.model_name}_regression_cdf.png'), dpi=150)
        plt.close(fig)

        # Scatter: true vs predicted
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(targets[:, 0], targets[:, 1], c='steelblue', alpha=0.3, s=10, label='True')
        ax.scatter(preds[:, 0],   preds[:, 1],   c='tomato',    alpha=0.3, s=10, label='Predicted')
        ax.set_xlabel('X coordinate (grid units)')
        ax.set_ylabel('Y coordinate (grid units)')
        ax.set_title(f'{self.__class__.__name__} — True vs Predicted Locations\n'
                     f'Mean Error: {mean_d:.3f}  Median: {median_d:.3f}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, f'{self.model_name}_regression_scatter.png'), dpi=150)
        plt.close(fig)

        print(f'  Plots saved to {log_dir}')
