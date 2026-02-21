import numpy as np
import os
from datetime import datetime
from data_loader import load_mnist_data
from experiments import experiment_baseline, experiment_hidden_size, experiment_learning_rate

def main():
    print("==========================================")
    print("Neural Network from Scratch - Homework 1")
    print("==========================================\n")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    try:
        X_train, y_train, X_test, y_test = load_mnist_data(
            "train-images.idx3-ubyte",
            "train-labels.idx1-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte"
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Input features: {X_train.shape[1]}")
    except FileNotFoundError as e:
        print(f"Error: MNIST data files not found!")
        print("Please download MNIST files and place them in the same directory as hw1.py")
        print("Required files:")
        print("  - train-images.idx3-ubyte")
        print("  - train-labels.idx1-ubyte")
        print("  - t10k-images.idx3-ubyte")
        print("  - t10k-labels.idx1-ubyte")
        return
    
    # Experiment 1: Baseline
    baseline_model, baseline_history = experiment_baseline(X_train, y_train, X_test, y_test, output_dir)
    
    # Experiment 2: Different hidden sizes
    hidden_sizes = [64, 128, 256]
    hidden_size_results = experiment_hidden_size(X_train, y_train, X_test, y_test, hidden_sizes, output_dir)
    
    # Experiment 3: Different learning rates
    learning_rates = [1.0, 0.1, 0.01, 0.001]
    learning_rate_results = experiment_learning_rate(X_train, y_train, X_test, y_test, learning_rates, output_dir)
    
    print("\n==========================================")
    print("All experiments completed!")
    print(f"All outputs saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  Baseline Experiment:")
    print("    - baseline_learning_curves.png")
    print("    - baseline_confusion_matrix.png")
    print("    - baseline_misclassified.png")
    print("    - baseline_weights.png")
    print("    - baseline_samples.png")
    print("    - baseline_per_class_accuracy.png")
    print("  Hidden Size Experiment:")
    print("    - hidden_size_comparison.png")
    print("    - hidden_size_final_metrics.png")
    print("    - hidden_64/128/256 confusion matrices and per-class accuracy")
    print("  Learning Rate Experiment:")
    print("    - learning_rate_comparison.png")
    print("    - learning_rate_final_metrics.png")
    print("    - lr_0.001/0.01/0.1 confusion matrices and per-class accuracy")
    print("==========================================")

if __name__ == "__main__":
    main()
