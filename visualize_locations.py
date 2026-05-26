# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei', 'SimSun', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from csi_dataset import CSIDataset, CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
from cnn_transformer_model import CNN_Transformer_Net
from base_model import plot_regression_report


MODEL_CLASSES = {
    'cnn': CNN_Net,
    'cnn_lstm': CNN_LSTM_Net,
    'cnn_transformer': CNN_Transformer_Net,
}


def create_location_grid(data_dir):
    dataset = CSIDataset(directory=data_dir, time_step=30, stride=1)

    x_coords = [loc[0] for loc in dataset.locations]
    y_coords = [loc[1] for loc in dataset.locations]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    plt.figure(figsize=(12, 10))
    plt.scatter(x_coords, y_coords, s=200, c='steelblue',
                edgecolors='black', linewidths=1.5, alpha=0.8)
    for x, y in dataset.locations:
        plt.annotate(f"({x:.1f},{y:.1f})", (x, y),
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
    plt.xlabel('X 坐标（米）')
    plt.ylabel('Y 坐标（米）')
    plt.title(f'WiFi 室内定位 — 采集点分布\n'
              f'共 {dataset.num_locations} 个位置')
    plt.savefig('location_grid.png', dpi=300)
    plt.show()

    points = np.array(list(zip(x_coords, y_coords)))
    try:
        vor = Voronoi(points)
        plt.figure(figsize=(12, 10))
        voronoi_plot_2d(vor, show_vertices=False, point_size=10)

        plt.scatter(x_coords, y_coords, s=150, c='steelblue',
                    edgecolors='black', linewidths=1, alpha=0.8)

        for x, y in dataset.locations:
            plt.annotate(f"({x:.1f},{y:.1f})", (x, y),
                         ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)
        plt.xlabel('X 坐标（米）')
        plt.ylabel('Y 坐标（米）')
        plt.title('采集点 Voronoi 区域划分')
        plt.savefig('location_voronoi.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"无法生成 Voronoi 图: {e}")

    print(f"\n共发现 {dataset.num_locations} 个采集位置:")
    for loc in dataset.locations:
        print(f"  ({loc[0]:.1f}, {loc[1]:.1f})")


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
    cls = MODEL_CLASSES.get(args.model_type)
    if cls is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    model = cls.load_from_checkpoint(
        args.model_path,
        time_step=data_module.time_step,
        num_subcarriers=data_module.num_subcarriers,
        in_channels=data_module.in_channels,
        reg_loss=args.reg_loss,
    )
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

    output_dir = os.path.join(os.getcwd(), 'results', f'{args.model_type}_eval')
    os.makedirs(output_dir, exist_ok=True)
    plot_regression_report(preds, targets, output_dir, args.model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize locations and evaluate regression models")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.getcwd(), "dataset"))

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to checkpoint. If provided, run model evaluation.")
    parser.add_argument("--model_type", type=str, default="cnn",
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--time_step", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--split_mode", type=str, default="by_location",
                        choices=["by_location", "random"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--reg_loss", type=str, default="smooth_l1",
                        choices=["smooth_l1", "mse", "mae"])

    args = parser.parse_args()

    if args.model_path:
        evaluate_model(args)
    else:
        create_location_grid(args.data_dir)
