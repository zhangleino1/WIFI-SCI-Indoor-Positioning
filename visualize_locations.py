# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from csi_dataset import CSIDataset
from scipy.spatial import Voronoi, voronoi_plot_2d


def create_location_grid(data_dir):
    """
    Visualize all measurement locations discovered in the dataset.
    """
    dataset = CSIDataset(directory=data_dir, time_step=30, stride=1)

    x_coords = [loc[0] for loc in dataset.locations]
    y_coords = [loc[1] for loc in dataset.locations]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # --- Scatter plot of locations ---
    plt.figure(figsize=(12, 10))
    plt.scatter(x_coords, y_coords, s=200, c='steelblue',
                edgecolors='black', linewidths=1.5, alpha=0.8)

    for x, y in dataset.locations:
        plt.annotate(f"({x},{y})", (x, y),
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
    plt.xlabel('X Coordinate (grid units)')
    plt.ylabel('Y Coordinate (grid units)')
    plt.title(f'WiFi Indoor Positioning — Measurement Locations\n'
              f'Total: {dataset.num_locations} locations')
    plt.savefig('location_grid.png', dpi=300)
    plt.show()

    # --- Voronoi diagram ---
    points = np.array(list(zip(x_coords, y_coords)))
    try:
        vor = Voronoi(points)
        plt.figure(figsize=(12, 10))
        voronoi_plot_2d(vor, show_vertices=False, point_size=10)

        plt.scatter(x_coords, y_coords, s=150, c='steelblue',
                    edgecolors='black', linewidths=1, alpha=0.8)

        for x, y in dataset.locations:
            plt.annotate(f"({x},{y})", (x, y),
                         ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)
        plt.xlabel('X Coordinate (grid units)')
        plt.ylabel('Y Coordinate (grid units)')
        plt.title('Approximate Location Voronoi Regions')
        plt.savefig('location_voronoi.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not create Voronoi diagram: {e}")

    # Print location listing
    print(f"\nFound {dataset.num_locations} unique locations:")
    for loc in dataset.locations:
        print(f"  ({loc[0]}, {loc[1]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize measurement locations for WiFi CSI indoor positioning")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.getcwd(), "dataset"))
    args = parser.parse_args()

    create_location_grid(args.data_dir)
