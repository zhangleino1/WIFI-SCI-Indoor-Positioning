import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from csi_dataset import CSIDataset
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d

def create_location_grid(data_dir):
    """
    Create a visual grid of all location classes
    """
    # Create a dataset to extract the location mapping
    dataset = CSIDataset(directory=data_dir, time_step=30, stride=1)
    
    # Extract all locations and their class indices
    locations = []
    for loc, idx in dataset.location_to_class.items():
        locations.append((int(loc[0]), int(loc[1]), idx))
    
    x_coords = [loc[0] for loc in locations]
    y_coords = [loc[1] for loc in locations]
    class_indices = [loc[2] for loc in locations]
    
    # Calculate grid dimensions
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Create a visually appealing scatter plot of the locations
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap
    cmap = plt.cm.viridis
    
    # Create the scatter plot with class indices
    scatter = plt.scatter(x_coords, y_coords, c=class_indices, s=200, cmap=cmap, 
                         edgecolors='black', linewidths=1.5, alpha=0.8)
    
    # Add labels to each point showing the class index and coordinates
    for i, (x, y, idx) in enumerate(locations):
        plt.annotate(f"Class {idx}\n({x},{y})", (x, y), 
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Expand the axis limits a bit for better visualization
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Class Index')
    
    # Set axis labels and title
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title(f'WiFi Indoor Positioning Location Classes\nTotal: {len(dataset.location_classes)} Classes')
    
    plt.savefig('location_classes_grid.png', dpi=300)
    plt.show()
    
    # Create a Voronoi diagram to show approximate decision boundaries
    plt.figure(figsize=(12, 10))
    
    # Convert locations to numpy array for Voronoi
    points = np.array(list(zip(x_coords, y_coords)))
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
        
        # Plot Voronoi diagram
        voronoi_plot_2d(vor, show_vertices=False, point_size=10)
        
        # Overlay the scatter plot
        scatter = plt.scatter(x_coords, y_coords, c=class_indices, s=150, cmap=cmap, 
                             edgecolors='black', linewidths=1, alpha=0.8)
        
        # Add labels to each point
        for i, (x, y, idx) in enumerate(locations):
            plt.annotate(f"{idx}", (x, y), 
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle='circle,pad=0.1', fc='white', alpha=0.7))
        
        plt.colorbar(scatter, label='Class Index')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title('Approximate Location Classification Boundaries')
        plt.savefig('location_voronoi.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not create Voronoi diagram: {e}")
    
    # Print location mapping
    print(f"\nFound {len(dataset.location_classes)} unique location classes")
    print("\nLocation to Class mapping:")
    for loc, idx in sorted(dataset.location_to_class.items(), key=lambda x: x[1]):
        print(f"Location ({loc[0]},{loc[1]}) -> Class {idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize location classes for WiFi CSI indoor positioning")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset", help="Directory containing CSI data")
    args = parser.parse_args()
    
    create_location_grid(args.data_dir)
