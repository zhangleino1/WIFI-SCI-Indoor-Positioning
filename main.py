# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from csi_dataset import CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
from cnn_transformer_model import CNN_Transformer_Net


def get_callbacks(args):
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        filename=f"{args.model_type}-best-{{epoch:02d}}-{{val_loss:.3f}}",
        save_top_k=1,
        mode='min',
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_cb, early_stop_cb, lr_monitor]


def get_model(args, num_classes: int):
    """Instantiate the selected model architecture."""
    num_subcarriers = 30
    common = dict(
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps,
        num_classes=num_classes,        # kept for API compat; not used by regression
        time_step=args.time_step,
        num_subcarriers=num_subcarriers,
        reg_loss=args.reg_loss,
    )
    if args.model_type == 'cnn':
        return CNN_Net(**common)
    elif args.model_type == 'cnn_lstm':
        return CNN_LSTM_Net(**common)
    elif args.model_type == 'cnn_transformer':
        return CNN_Transformer_Net(**common)
    raise ValueError(f"Unknown model_type: {args.model_type}")


def train(args):
    data_module = CSIDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_step=args.time_step,
        data_dir=args.data_dir,
        stride=args.stride,
    )
    model   = get_model(args, data_module.num_classes)
    logger  = TensorBoardLogger('./logs', name=args.model_type)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using accelerator: {accelerator}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == 'gpu' else None,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=get_callbacks(args),
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        min_steps=args.min_steps,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def test(args):
    data_module = CSIDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_step=args.time_step,
        data_dir=args.data_dir,
        stride=args.stride,
    )
    logger = TensorBoardLogger('./logs', name=f'{args.model_type}_test')

    load_kwargs = dict(
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps,
        num_classes=data_module.num_classes,
        time_step=args.time_step,
        num_subcarriers=30,
        reg_loss=args.reg_loss,
    )

    if args.model_type == 'cnn':
        model = CNN_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    elif args.model_type == 'cnn_lstm':
        model = CNN_LSTM_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    elif args.model_type == 'cnn_transformer':
        model = CNN_Transformer_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == 'gpu' else None,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=get_callbacks(args),
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        min_steps=args.min_steps,
    )
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WiFi CSI Indoor Positioning — Regression Training & Evaluation')

    # ---- data ----
    parser.add_argument('--data_dir',    type=str, default=os.path.join(os.getcwd(), 'dataset'))
    parser.add_argument('--time_step',   type=int, default=15,
                        help='Number of consecutive CSI rows per sample')
    parser.add_argument('--stride',      type=int, default=2,
                        help='Sliding window step between samples')
    parser.add_argument('--num_workers', type=int, default=8)

    # ---- model ----
    parser.add_argument('--model_type',  type=str, default='cnn',
                        choices=['cnn', 'cnn_lstm', 'cnn_transformer'])
    parser.add_argument('--reg_loss',    type=str, default='smooth_l1',
                        choices=['smooth_l1', 'mse', 'mae'],
                        help=('Regression loss function. '
                              'smooth_l1 (Huber): robust to outliers, recommended default. '
                              'mse: penalises large errors more; sensitive to outliers. '
                              'mae (L1): most robust; slower convergence.'))

    # ---- training ----
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--lr_factor',   type=float, default=0.1,
                        help='LR reduction factor for ReduceLROnPlateau')
    parser.add_argument('--lr_patience', type=int,   default=10,
                        help='Epochs without improvement before LR reduction')
    parser.add_argument('--lr_eps',      type=float, default=1e-6)
    parser.add_argument('--max_epochs',  type=int,   default=120)
    parser.add_argument('--min_epochs',  type=int,   default=10)
    parser.add_argument('--min_steps',   type=int,   default=5)
    parser.add_argument('--fast_dev_run', default=False, type=bool,
                        help='Run one batch for quick debugging')

    # ---- mode ----
    parser.add_argument('--mode',       type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--cpt_path',   type=str,
                        default=os.path.join(os.getcwd(),
                                             'logs/cnn/version_0/checkpoints/last.ckpt'),
                        help='Checkpoint path (required for --mode test)')

    args = parser.parse_args()
    if args.mode == 'test':
        test(args)
    else:
        train(args)
