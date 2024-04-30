import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from csi_dataset import CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
import torch
import os


def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=f"{args.model_type}-best-{{epoch:02d}}-{{val_loss:.3f}}",
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

    return [checkpoint_callback, early_stopping, lr_monitor]

def train(args):
    # 设置数据模块
    data_module = CSIDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                time_step=args.time_step, data_dir=args.data_dir, stride=args.stripe)

    # 设置模型
    model = get_model(args)


    # 设置日志
    logger = TensorBoardLogger("./logs",  name=f"{args.model_type}")

    # Check if GPU is available and set the accelerator accordingly
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(accelerator)

    # 训练器
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
    # 训练模型
    trainer.fit(model, datamodule=data_module)
     # Test the model
    trainer.test(model,data_module)

def get_model(args):
    if args.model_type == 'cnn':
        return CNN_Net(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_eps=args.lr_eps)

    elif args.model_type == 'cnn_lstm':
        return CNN_LSTM_Net(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_eps=args.lr_eps)
    else:
        raise ValueError("Invalid model type specified")

def test(args):
    logger = TensorBoardLogger(
            save_dir="./logs",
            name=f"{args.model_type}test"
        )

        # 将 Namespace 对象转换为字典
    hparams_dict = vars(args)
    if args.model_type == 'cnn':
        model = CNN_Net.load_from_checkpoint(args.cpt_path,lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_eps=args.lr_eps)
    elif args.model_type == 'cnn_lstm':
        model = CNN_LSTM_Net.load_from_checkpoint(args.cpt_path,lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, lr_eps=args.lr_eps)
    else:
        raise ValueError("Invalid model type specified")
    data_module = CSIDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                time_step=args.time_step, data_dir=args.data_dir, stride=args.stripe)
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
    trainer.test(model,datamodule=data_module)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural network on WiFi CSI data")
    parser.add_argument("--batch_size", type=int, default=128, help="Input batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument("--lr_eps", type=float, default=1e-6, help="Epsilon for learning rate reduction")
    parser.add_argument("--time_step", type=int, default=30, help="Time steps to use for each sample")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset", help="Directory for output")
    parser.add_argument('--min_epochs', default=10, type=int)
    parser.add_argument('--max_epochs', default=120, type=int)
    parser.add_argument('--min_steps', type=int, default=5)
    parser.add_argument('--fast_dev_run', default=False, type=bool)
    parser.add_argument('--stripe', type=int, default=1)
    parser.add_argument('--accelerator', default="gpu", type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--mode', choices=['test','train'], type=str,default='train')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'cnn_lstm'],
                        help='Model type to train/test')
    parser.add_argument('--cpt_path', default=os.getcwd() + '/checkpoints/last.ckpt', type=str)
    
    args = parser.parse_args()
    if args.mode == 'test':
        test(args)
    else:
        train(args)

