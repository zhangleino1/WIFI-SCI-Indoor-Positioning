import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from csi_dataset import CSIDataModule
from csi_net_model import CSINet
import torch
import os


def main(args):
    # 设置数据模块
    data_module = CSIDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                time_step=args.time_step, data_dir=args.data_dir)

    # 设置模型
    model = CSINet(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_eps=args.lr_eps)

    # 设置检查点保存和学习率日志
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='csi-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 设置日志
    logger = TensorBoardLogger("./logs", name="csi_net")

    # Check if GPU is available and set the accelerator accordingly
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # 训练器
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == 'gpu' else None,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        fast_dev_run=args.fast_dev_run, 
        min_steps=args.min_steps,
    )
    # 训练模型
    trainer.fit(model, datamodule=data_module)
     # Test the model
    trainer.test(model,data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural network on WiFi CSI data")
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument("--lr_eps", type=float, default=1e-6, help="Epsilon for learning rate reduction")
    parser.add_argument("--time_step", type=int, default=30, help="Time steps to use for each sample")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset", help="Directory for output")
    parser.add_argument('--min_epochs', default=100, type=int)
    parser.add_argument('--max_epochs', default=120, type=int)
    parser.add_argument('--min_steps', type=int, default=5)
    parser.add_argument('--fast_dev_run', default=False, type=bool)
    args = parser.parse_args()
    main(args)
