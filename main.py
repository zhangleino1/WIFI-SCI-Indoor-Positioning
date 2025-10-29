# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from csi_dataset import CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
from cnn_transformer_model import CNN_Transformer_Net # Import the new model
import torch
import os


def get_callbacks(args):
    monitor_metric = 'val_loss'
    mode = 'min'  # Changed from 'max' to 'min'
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=f"{args.model_type}-best-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        save_top_k=1,
        mode=mode,  # This will now use 'min'
        save_last=True
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode=mode  # This will now use 'min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return [checkpoint_callback, early_stopping, lr_monitor]

def train(args):
    # 设置数据模块
    data_module = CSIDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                time_step=args.time_step, data_dir=args.data_dir, stride=args.stride)

    # 设置模型 - 传递分类数量
    model = get_model(args, data_module.num_classes)


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

def get_model(args, num_classes):
    num_subcarriers = 30 # Based on dataset analysis
    if args.model_type == 'cnn':
        return CNN_Net(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, 
                       lr_eps=args.lr_eps, num_classes=num_classes, 
                       time_step=args.time_step, num_subcarriers=num_subcarriers)

    elif args.model_type == 'cnn_lstm':
        return CNN_LSTM_Net(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience, 
                            lr_eps=args.lr_eps, num_classes=num_classes,
                            time_step=args.time_step, num_subcarriers=num_subcarriers)
    elif args.model_type == 'cnn_transformer': # Add new model type
        return CNN_Transformer_Net(lr=args.lr, lr_factor=args.lr_factor, lr_patience=args.lr_patience,
                                   lr_eps=args.lr_eps, num_classes=num_classes,
                                   time_step=args.time_step, num_subcarriers=num_subcarriers)
                                   # d_model, nhead, num_encoder_layers, dim_feedforward will use defaults from __init__
    else:
        raise ValueError("Invalid model type specified")

def test(args):
    # 首先创建数据模块以获取类别数量
    data_module = CSIDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                time_step=args.time_step, data_dir=args.data_dir, stride=args.stride)
    
    logger = TensorBoardLogger(
            save_dir="./logs",
            name=f"{args.model_type}_test" # Underscore for clarity
        )

    # Define num_subcarriers, consistent with model training
    num_subcarriers = 30 

    # For loading from checkpoint, ensure all necessary hparams are passed if they 
    # were not saved with save_hyperparameters() or if you need to override them.
    # Since models now use save_hyperparameters(), these will serve as defaults if not in checkpoint,
    # or override if values in checkpoint are different and that's desired.
    # However, num_classes, time_step, num_subcarriers are structural and should match the trained model.
    # So, it's good practice to pass them, especially if they might vary.
    
    # It's generally better to load hparams from the checkpoint if possible,
    # but for structural params like num_classes, time_step, num_subcarriers,
    # they must be correct for the architecture.
    
    # Let's assume the checkpoint has the correct hparams saved by save_hyperparameters().
    # If not, or to be explicit:
    hparams_override = {
        'lr': args.lr,
        'lr_factor': args.lr_factor,
        'lr_patience': args.lr_patience,
        'lr_eps': args.lr_eps,
        'num_classes': data_module.num_classes,
        'time_step': args.time_step,
        'num_subcarriers': num_subcarriers
    }

    if args.model_type == 'cnn':
        model = CNN_Net.load_from_checkpoint(
            args.cpt_path,
            # Pass hparams that might need overriding or were not saved.
            # If save_hyperparameters() was used consistently, only map_location might be needed.
            # For safety, passing them all if they are part of args.
            lr=args.lr, 
            lr_factor=args.lr_factor, 
            lr_patience=args.lr_patience, 
            lr_eps=args.lr_eps,
            num_classes=data_module.num_classes,
            time_step=args.time_step, # Crucial for model structure
            num_subcarriers=num_subcarriers # Crucial for model structure
        )
    elif args.model_type == 'cnn_lstm':
        model = CNN_LSTM_Net.load_from_checkpoint(
            args.cpt_path,
            lr=args.lr, 
            lr_factor=args.lr_factor, 
            lr_patience=args.lr_patience, 
            lr_eps=args.lr_eps,
            num_classes=data_module.num_classes,
            time_step=args.time_step, # Crucial for model structure
            num_subcarriers=num_subcarriers # Crucial for model structure
        )
    elif args.model_type == 'cnn_transformer': # Add new model type
        model = CNN_Transformer_Net.load_from_checkpoint(
            args.cpt_path,
            lr=args.lr, 
            lr_factor=args.lr_factor, 
            lr_patience=args.lr_patience, 
            lr_eps=args.lr_eps,
            num_classes=data_module.num_classes,
            time_step=args.time_step,
            num_subcarriers=num_subcarriers
            # d_model, nhead, etc., will be loaded from checkpoint hparams if saved by save_hyperparameters()
        )
    else:
        raise ValueError("Invalid model type specified")
    
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
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument("--lr_eps", type=float, default=1e-6, help="Epsilon for learning rate reduction")
    parser.add_argument("--time_step", type=int, default=15, help="Time steps to use for each sample")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset", help="Directory for output")
    parser.add_argument('--min_epochs', default=10, type=int)
    parser.add_argument('--max_epochs', default=120, type=int)
    parser.add_argument('--min_steps', type=int, default=5)
    parser.add_argument('--fast_dev_run', default=False, type=bool)
    parser.add_argument('--stride', type=int, default=2, help="Stride for sliding window")
    parser.add_argument('--accelerator', default="gpu", type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--mode', choices=['test','train'], type=str,default='train')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'cnn_lstm', 'cnn_transformer'], # Added 'cnn_transformer'
                        help='Model type to train/test')
    parser.add_argument('--cpt_path', default=os.getcwd() + '/logs/cnn/version_0/checkpoints/cnn-best-epoch=00-val_loss=4.558.ckpt', type=str)

    
    args = parser.parse_args()
    if args.mode == 'test':
        test(args)
    else:
        train(args)

