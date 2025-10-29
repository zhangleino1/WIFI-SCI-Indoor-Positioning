# 作者：程序员石磊，盗用卖钱可耻，在github即可搜到
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
from csi_dataset import CSIDataset, CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net

def analyze_spatial_confusion(model_path, data_dir, model_type='cnn_lstm', batch_size=32):
    """
    Analyze how confusion relates to physical distance between locations
    """
    # Setup the data module
    data_module = CSIDataModule(
        batch_size=batch_size, 
        num_workers=4, 
        time_step=15, 
        data_dir=data_dir,
        stride=1
    )
    
    # Get the number of classes and mapping
    num_classes = data_module.num_classes
    loc_to_class = data_module.dataset.location_to_class
    class_to_loc = {v: k for k, v in loc_to_class.items()}
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    if model_type == 'cnn':
        model = CNN_Net.load_from_checkpoint(
            model_path,
            lr=0.0001,
            lr_factor=0.1,
            lr_patience=10,
            lr_eps=1e-6,
            num_classes=num_classes
        )
    elif model_type == 'cnn_lstm':
        model = CNN_LSTM_Net.load_from_checkpoint(
            model_path,
            lr=0.0001,
            lr_factor=0.1,
            lr_patience=10,
            lr_eps=1e-6,
            num_classes=num_classes
        )
    
    model = model.to(device)
    model.eval()
    
    # Get the test dataloader
    test_loader = data_module.test_dataloader()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    
    # Run prediction
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate physical distance matrix between all class locations
    distance_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        loc_i = class_to_loc[i]
        x1, y1 = int(loc_i[0]), int(loc_i[1])
        
        for j in range(num_classes):
            loc_j = class_to_loc[j]
            x2, y2 = int(loc_j[0]), int(loc_j[1])
            
            # Calculate Euclidean distance
            distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Create data for the error vs distance analysis
    errors = []
    distances = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Skip the diagonal
                # Number of times class i was predicted as class j
                error_count = cm[i, j]
                
                # The physical distance between locations
                dist = distance_matrix[i, j]
                
                # Add to lists
                errors.append(error_count)
                distances.append(dist)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(distances, errors, alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(distances, errors, 1)
    p = np.poly1d(z)
    plt.plot(sorted(distances), p(sorted(distances)), "r--", linewidth=2)
    
    plt.xlabel('Physical Distance Between Locations (meters)')
    plt.ylabel('Number of Confusion Errors')
    plt.title(f'{model_type.upper()} - Relationship Between Physical Distance and Classification Errors')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_type}_distance_vs_errors.png', dpi=300)
    plt.show()
    
    # Create a heatmap of the normalized confusion matrix with physical distances
    plt.figure(figsize=(14, 12))
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create DataFrame for better plot labels
    class_labels = [f"({class_to_loc[i][0]},{class_to_loc[i][1]})" for i in range(num_classes)]
    
    # Create heatmap with annotations showing physical distance
    ax = sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    
    # Add distance annotations to cells with significant errors
    for i in range(num_classes):
        for j in range(num_classes):
            # Skip the diagonal and cells with very small values
            if i != j and cm_norm[i, j] > 0.05:  # Only show for significant errors (>5%)
                ax.text(j + 0.5, i + 0.5, f'd={distance_matrix[i, j]:.1f}m', 
                        ha='center', va='center', fontsize=8, 
                        color='white' if cm_norm[i, j] > 0.3 else 'black')
    
    plt.xlabel('Predicted Location')
    plt.ylabel('True Location')
    plt.title(f'{model_type.upper()} - Confusion Matrix with Physical Distances (m)')
    plt.tight_layout()
    plt.savefig(f'{model_type}_confusion_with_distances.png', dpi=300)
    plt.show()
    
    # Create a histogram of error distances
    error_distances = []
    for i, (true_class, pred_class) in enumerate(zip(all_targets, all_preds)):
        if true_class != pred_class:  # If prediction was wrong
            error_dist = distance_matrix[true_class, pred_class]
            error_distances.append(error_dist)
    
    if error_distances:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(error_distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
        
        # Calculate mean and median
        mean_dist = np.mean(error_distances)
        median_dist = np.median(error_distances)
        
        # Add vertical lines for mean and median
        plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dist:.2f}m')
        plt.axvline(median_dist, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_dist:.2f}m')
        
        plt.xlabel('Distance Between True and Predicted Locations (meters)')
        plt.ylabel('Count')
        plt.title(f'{model_type.upper()} - Distribution of Error Distances')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_type}_error_distance_histogram.png', dpi=300)
        plt.show()
    else:
        print("No classification errors to plot!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spatial confusion in indoor positioning")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint",default=os.getcwd()+"/logs/cnn_lstm/version_2/checkpoints/cnn_lstm-best-epoch=22-val_loss=4.505.ckpt")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset", help="Directory containing CSI data")
    parser.add_argument("--model_type", type=str, default='cnn_lstm', choices=['cnn', 'cnn_lstm'], help="Model type to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    analyze_spatial_confusion(args.model_path, args.data_dir, args.model_type, args.batch_size)
