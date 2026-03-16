# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from csi_dataset import CSIDataModule
from cnn_lstm_net_model import CNN_LSTM_Net
from cnn_net_model import CNN_Net
from cnn_transformer_model import CNN_Transformer_Net


def load_model(model_path, model_type, data_module, reg_loss):
    load_kwargs = dict(
        lr=0.001,
        lr_factor=0.1,
        lr_patience=10,
        lr_eps=1e-6,
        time_step=data_module.time_step,
        num_subcarriers=data_module.num_subcarriers,
        reg_loss=reg_loss,
    )

    if model_type == 'cnn':
        return CNN_Net.load_from_checkpoint(model_path, **load_kwargs)
    if model_type == 'cnn_lstm':
        return CNN_LSTM_Net.load_from_checkpoint(model_path, **load_kwargs)
    if model_type == 'cnn_transformer':
        return CNN_Transformer_Net.load_from_checkpoint(model_path, **load_kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


def plot_regression_results(preds, targets, distances, output_dir, model_type):
    sorted_distances = np.sort(distances)
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_distances, cdf, linewidth=2)
    ax.set_xlabel('Distance Error (grid units)')
    ax.set_ylabel('CDF')
    ax.set_title(f'{model_type.upper()} Regression Error CDF')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{model_type}_regression_cdf.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets[:, 0], targets[:, 1], c='steelblue', alpha=0.4, s=12, label='True')
    ax.scatter(preds[:, 0], preds[:, 1], c='tomato', alpha=0.4, s=12, label='Predicted')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'{model_type.upper()} True vs Predicted Locations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{model_type}_regression_scatter.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distances, bins=20, alpha=0.8, color='tab:blue', edgecolor='black')
    ax.set_xlabel('Distance Error (grid units)')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_type.upper()} Regression Error Histogram')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{model_type}_regression_hist.png'), dpi=150)
    plt.close(fig)


def evaluate_model(args):
    data_module = CSIDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_step=args.time_step,
        data_dir=args.data_dir,
        stride=args.stride,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, args.model_type, data_module, args.reg_loss)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_module.test_dataloader():
            data = data.to(device)
            preds = model(data)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    distances = np.sqrt(((preds - targets) ** 2).sum(axis=1))

    mean_dist = distances.mean()
    median_dist = np.median(distances)
    within_1 = (distances <= 1.0).mean() * 100
    within_2 = (distances <= 2.0).mean() * 100
    within_3 = (distances <= 3.0).mean() * 100

    print(f'Mean Distance Error   : {mean_dist:.4f}')
    print(f'Median Distance Error : {median_dist:.4f}')
    print(f'Within 1 unit         : {within_1:.1f}%')
    print(f'Within 2 units        : {within_2:.1f}%')
    print(f'Within 3 units        : {within_3:.1f}%')

    output_dir = os.path.join(os.getcwd(), 'results', f'{args.model_type}_eval')
    os.makedirs(output_dir, exist_ok=True)
    plot_regression_results(preds, targets, distances, output_dir, args.model_type)
    print(f'Plots saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate regression models for WiFi CSI indoor positioning')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(os.getcwd(), 'logs/cnn/version_0/checkpoints/last.ckpt'),
        help='Path to trained model checkpoint',
    )
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'dataset'))
    parser.add_argument(
        '--model_type',
        type=str,
        default='cnn',
        choices=['cnn', 'cnn_lstm', 'cnn_transformer'],
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--time_step', type=int, default=15)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument(
        '--split_mode',
        type=str,
        default='by_location',
        choices=['by_location', 'random'],
    )
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument(
        '--reg_loss',
        type=str,
        default='smooth_l1',
        choices=['smooth_l1', 'mse', 'mae'],
    )

    evaluate_model(parser.parse_args())
