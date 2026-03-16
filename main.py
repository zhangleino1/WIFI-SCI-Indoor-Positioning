# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from csi_dataset import CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
from cnn_transformer_model import CNN_Transformer_Net
import torch


def get_callbacks(args):
    monitor_metric = 'val_loss'
    mode = 'min'

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=f"{args.model_type}-best-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        save_top_k=1,
        mode=mode,
        save_last=True
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode=mode
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_callback, early_stopping, lr_monitor]


def get_model(args, num_classes):
    num_subcarriers = 30
    common = dict(
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps,
        num_classes=num_classes,
        time_step=args.time_step,
        num_subcarriers=num_subcarriers,
        task=args.task,
    )
    if args.model_type == 'cnn':
        return CNN_Net(**common)
    elif args.model_type == 'cnn_lstm':
        return CNN_LSTM_Net(**common)
    elif args.model_type == 'cnn_transformer':
        return CNN_Transformer_Net(**common)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


def train(args):
    data_module = CSIDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_step=args.time_step,
        data_dir=args.data_dir,
        stride=args.stride,
        task=args.task,
    )

    model = get_model(args, data_module.num_classes)

    logger = TensorBoardLogger("./logs", name=f"{args.model_type}")

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(accelerator)

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
    trainer.test(model, data_module)


def test(args):
    data_module = CSIDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_step=args.time_step,
        data_dir=args.data_dir,
        stride=args.stride,
        task=args.task,
    )

    logger = TensorBoardLogger(save_dir="./logs", name=f"{args.model_type}_test")

    num_subcarriers = 30
    load_kwargs = dict(
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_eps=args.lr_eps,
        num_classes=data_module.num_classes,
        time_step=args.time_step,
        num_subcarriers=num_subcarriers,
        task=args.task,
    )

    if args.model_type == 'cnn':
        model = CNN_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    elif args.model_type == 'cnn_lstm':
        model = CNN_LSTM_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    elif args.model_type == 'cnn_transformer':
        model = CNN_Transformer_Net.load_from_checkpoint(args.cpt_path, **load_kwargs)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

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
    parser = argparse.ArgumentParser(description="Train a neural network on WiFi CSI data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_eps", type=float, default=1e-6)
    parser.add_argument("--time_step", type=int, default=15)
    parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/dataset")
    parser.add_argument('--min_epochs', default=10, type=int)
    parser.add_argument('--max_epochs', default=120, type=int)
    parser.add_argument('--min_steps', type=int, default=5)
    parser.add_argument('--fast_dev_run', default=False, type=bool)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--accelerator', default="gpu", type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--mode', choices=['test', 'train'], type=str, default='train')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'cnn_lstm', 'cnn_transformer'])
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help=(
                            "Task type. 'classification' treats each (x,y) location as a separate class "
                            "and reports accuracy. 'regression' predicts (x,y) coordinates directly and "
                            "reports mean Euclidean distance error. Regression is recommended when "
                            "classification accuracy is low due to many location classes."
                        ))
    parser.add_argument('--cpt_path', default=os.getcwd() + '/logs/cnn/version_0/checkpoints/last.ckpt', type=str)

    args = parser.parse_args()
    if args.mode == 'test':
        test(args)
    else:
        train(args)
