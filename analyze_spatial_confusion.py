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
    raise ValueError(f'Unsupported model type: {model_type}')


def analyze_spatial_confusion(args):
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
    model = load_model(args.model_path, args.model_type, data_module, args.reg_loss).to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in data_module.test_dataloader():
            preds = model(data.to(device))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    errors = preds - targets
    distances = np.sqrt((errors ** 2).sum(axis=1))
    true_radius = np.sqrt((targets ** 2).sum(axis=1))

    output_dir = os.path.join(os.getcwd(), 'results', f'{args.model_type}_spatial_analysis')
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_radius, distances, alpha=0.5, s=18)
    if len(true_radius) > 1:
        coeffs = np.polyfit(true_radius, distances, deg=1)
        line_x = np.linspace(true_radius.min(), true_radius.max(), 200)
        line_y = coeffs[0] * line_x + coeffs[1]
        ax.plot(line_x, line_y, 'r--', linewidth=2, label='Linear trend')
        ax.legend()
    ax.set_xlabel('Distance of True Position from Origin')
    ax.set_ylabel('Prediction Error Distance')
    ax.set_title(f'{args.model_type.upper()} Spatial Error vs True Position Radius')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{args.model_type}_error_vs_radius.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(distances, bins=20, alpha=0.8, color='tab:blue', edgecolor='black')
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean {mean_dist:.2f}')
    ax.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median {median_dist:.2f}')
    ax.set_xlabel('Prediction Error Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'{args.model_type.upper()} Regression Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{args.model_type}_error_histogram.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    q = ax.quiver(
        targets[:, 0],
        targets[:, 1],
        errors[:, 0],
        errors[:, 1],
        distances,
        angles='xy',
        scale_units='xy',
        scale=1,
        cmap='viridis',
        width=0.004,
    )
    plt.colorbar(q, ax=ax, label='Error Distance')
    ax.scatter(targets[:, 0], targets[:, 1], c='black', s=10, alpha=0.35)
    ax.set_xlabel('True X')
    ax.set_ylabel('True Y')
    ax.set_title(f'{args.model_type.upper()} Error Vectors by Position')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{args.model_type}_error_vectors.png'), dpi=150)
    plt.close(fig)

    print(f'Mean Distance Error   : {mean_dist:.4f}')
    print(f'Median Distance Error : {median_dist:.4f}')
    print(f'Plots saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze spatial error patterns for regression models')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(os.getcwd(), 'logs/cnn_lstm/version_0/checkpoints/last.ckpt'),
        help='Path to trained model checkpoint',
    )
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'dataset'))
    parser.add_argument(
        '--model_type',
        type=str,
        default='cnn_lstm',
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

    analyze_spatial_confusion(parser.parse_args())
