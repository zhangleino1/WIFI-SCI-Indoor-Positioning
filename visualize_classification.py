import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from csi_dataset import CSIDataset, CSIDataModule
from cnn_net_model import CNN_Net
from cnn_lstm_net_model import CNN_LSTM_Net
import pandas as pd

def visualize_confusion_matrix(cm, class_names, title, normalize=False, save_path=None):
    """
    Generate a confusion matrix visualization
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Location')
    plt.ylabel('True Location')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def evaluate_model(model_path, data_dir, model_type='cnn_lstm', batch_size=32):
    """
    Evaluate a trained model and visualize its performance
    """
    # Setup the data module
    data_module = CSIDataModule(
        batch_size=batch_size, 
        num_workers=4, 
        time_step=30, 
        data_dir=data_dir,
        stride=1
    )
    
    # Get the number of classes
    num_classes = data_module.num_classes
    
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    # Get the test dataloader
    test_loader = data_module.test_dataloader()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    all_correct = []  # To store whether each prediction was correct
    loc_to_class = data_module.dataset.location_to_class
    class_to_loc = {v: k for k, v in loc_to_class.items()}
    
    # Run prediction
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Record correct predictions
            correct = (preds == targets).cpu().numpy()
            all_correct.extend(correct)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    # Generate class names based on locations
    class_names = [f"({class_to_loc[i][0]},{class_to_loc[i][1]})" for i in range(num_classes)]
    
    # Visualize confusion matrix
    visualize_confusion_matrix(
        cm, 
        class_names, 
        f"{model_type.upper()} Confusion Matrix (Accuracy: {accuracy:.4f})",
        normalize=True,
        save_path=f"{model_type}_confusion_matrix_norm.png"
    )
    
    # Also show non-normalized confusion matrix
    visualize_confusion_matrix(
        cm, 
        class_names, 
        f"{model_type.upper()} Confusion Matrix - Raw Counts",
        normalize=False,
        save_path=f"{model_type}_confusion_matrix_raw.png"
    )
    
    # Calculate accuracy per location
    class_accuracies = {}
    for i in range(num_classes):
        # Get indices where true class is i
        indices = np.where(np.array(all_targets) == i)[0]
        if len(indices) > 0:
            # Calculate accuracy for this class
            class_acc = sum([all_correct[j] for j in indices]) / len(indices)
            class_accuracies[i] = class_acc
    
    # Visualize accuracy per location
    plt.figure(figsize=(14, 8))
    location_coords = [(int(class_to_loc[i][0]), int(class_to_loc[i][1])) for i in range(num_classes)]
    x_coords = [loc[0] for loc in location_coords]
    y_coords = [loc[1] for loc in location_coords]
    accuracies = [class_accuracies.get(i, 0) for i in range(num_classes)]
    
    # Create scatter plot with location accuracy
    scatter = plt.scatter(x_coords, y_coords, c=accuracies, s=200, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add location labels
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"({x},{y})\n{accuracies[i]:.2f}", (x, y), 
                     ha='center', va='center', fontsize=9, 
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.colorbar(scatter, label='Accuracy')
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'{model_type.upper()} Classification Accuracy by Location')
    plt.savefig(f'{model_type}_location_accuracy.png', dpi=300)
    plt.show()
    
    # Calculate error distribution
    error_counts = {}
    for true_class, pred_class in zip(all_targets, all_preds):
        if true_class != pred_class:  # If prediction was wrong
            true_loc = class_to_loc[true_class]
            pred_loc = class_to_loc[pred_class]
            
            # Calculate Euclidean distance between true and predicted locations
            x1, y1 = int(true_loc[0]), int(true_loc[1])
            x2, y2 = int(pred_loc[0]), int(pred_loc[1])
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Round distance to nearest 0.5
            distance = round(distance * 2) / 2
            
            if distance not in error_counts:
                error_counts[distance] = 0
            error_counts[distance] += 1
    
    # Plot error distribution
    if error_counts:
        plt.figure(figsize=(10, 6))
        distances = sorted(error_counts.keys())
        counts = [error_counts[d] for d in distances]
        
        # Calculate CDF
        total_errors = sum(counts)
        cumulative = np.cumsum(counts) / total_errors if total_errors > 0 else np.zeros_like(counts)
        
        plt.bar(distances, counts, alpha=0.7, label='Error Count')
        plt.plot(distances, cumulative, 'ro-', linewidth=2, markersize=8, label='CDF')
        
        plt.xlabel('Distance Error (meters)')
        plt.ylabel('Count / Cumulative Probability')
        plt.title(f'{model_type.upper()} Distance Error Distribution')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_type}_error_distribution.png', dpi=300)
        plt.show()
    else:
        print("No classification errors to plot!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize classification metrics for indoor positioning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default=os.getcwd()+"/dataset_test", help="Directory containing CSI data")
    parser.add_argument("--model_type", type=str, default='cnn_lstm', choices=['cnn', 'cnn_lstm'], help="Model type to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir, args.model_type, args.batch_size)
