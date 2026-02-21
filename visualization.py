import numpy as np
import matplotlib.pyplot as plt
from utils import compute_accuracy

def plot_confusion_matrix(model, X_test, y_test, output_dir, experiment_name):
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    num_classes = 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_test, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{experiment_name} - Confusion Matrix')
    
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_matrix[i, j]), 
                    ha='center', va='center',
                    color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black')
    
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_confusion_matrix.png")

def plot_misclassified_examples(model, X_test, y_test, output_dir, experiment_name, num_examples=20):
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    misclassified_indices = np.where(pred_labels != y_test)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
    
    sample_indices = misclassified_indices[:num_examples]
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(f'{experiment_name} - Misclassified Examples', fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        if idx >= len(sample_indices):
            ax.axis('off')
            continue
        
        img_idx = sample_indices[idx]
        img = X_test[img_idx].reshape(28, 28)
        true_label = y_test[img_idx]
        pred_label = pred_labels[img_idx]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_misclassified.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_misclassified.png")

def plot_weight_visualization(model, output_dir, experiment_name):
    W1 = model.W1[:-1, :]  # Exclude bias
    
    num_hidden = min(W1.shape[1], 100)
    num_cols = 10
    num_rows = (num_hidden + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 1.5))
    fig.suptitle(f'{experiment_name} - First Layer Weights', fontsize=14)
    
    for idx in range(num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        if idx < num_hidden:
            weight = W1[:, idx].reshape(28, 28)
            ax.imshow(weight, cmap='RdBu', vmin=-0.1, vmax=0.1)
            ax.set_title(f'H{idx}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_weights.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_weights.png")

def plot_sample_predictions(model, X_test, y_test, output_dir, experiment_name, num_samples=20):
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Get random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(f'{experiment_name} - Sample Predictions', fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        img_idx = sample_indices[idx]
        img = X_test[img_idx].reshape(28, 28)
        true_label = y_test[img_idx]
        pred_label = pred_labels[img_idx]
        confidence = predictions[img_idx, pred_label]
        
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}', 
                    fontsize=9, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_samples.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_samples.png")

def plot_per_class_accuracy(model, X_test, y_test, output_dir, experiment_name):
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    num_classes = 10
    class_accuracies = []
    
    for class_idx in range(num_classes):
        class_mask = y_test == class_idx
        class_predictions = pred_labels[class_mask]
        class_true = y_test[class_mask]
        accuracy = np.mean(class_predictions == class_true)
        class_accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(num_classes), class_accuracies, color='steelblue', edgecolor='black')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy')
    plt.title(f'{experiment_name} - Per-Class Accuracy')
    plt.xticks(range(num_classes))
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_per_class_accuracy.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_per_class_accuracy.png")

def plot_loss_comparison_bar(results, output_dir, comparison_name, parameter_name):
    param_values = list(results.keys())
    final_train_losses = [results[p]['train_losses'][-1] for p in param_values]
    final_test_losses = [results[p]['test_losses'][-1] for p in param_values]
    final_train_accs = [results[p]['train_accuracies'][-1] for p in param_values]
    final_test_accs = [results[p]['test_accuracies'][-1] for p in param_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(param_values))
    width = 0.35
    
    # Loss comparison
    axes[0].bar(x - width/2, final_train_losses, width, label='Train Loss', color='skyblue')
    axes[0].bar(x + width/2, final_test_losses, width, label='Test Loss', color='orange')
    axes[0].set_xlabel(parameter_name)
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Final Loss Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(param_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[1].bar(x - width/2, final_train_accs, width, label='Train Accuracy', color='lightgreen')
    axes[1].bar(x + width/2, final_test_accs, width, label='Test Accuracy', color='salmon')
    axes[1].set_xlabel(parameter_name)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Final Accuracy Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_values)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{comparison_name}_final_metrics.png', dpi=150)
    plt.close()
    print(f"Saved: {comparison_name}_final_metrics.png")
