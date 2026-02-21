import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from train import train_network
from visualization import (plot_confusion_matrix, plot_misclassified_examples, 
                          plot_weight_visualization, plot_sample_predictions,
                          plot_per_class_accuracy, plot_loss_comparison_bar)

def plot_learning_curves(history, experiment_name, output_dir='output'):
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['test_losses'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{experiment_name} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_accuracies'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{experiment_name} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}_learning_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {experiment_name}_learning_curves.png")

def experiment_baseline(X_train, y_train, X_test, y_test, output_dir='output'):
    print("\n=== Baseline Experiment ===")
    print("Training with default parameters...")
    
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 10
    epochs = 50
    batch_size = 32
    learning_rate = 0.01
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    history = train_network(model, X_train, y_train, X_test, y_test, 
                          epochs, batch_size, learning_rate)
    
    plot_learning_curves(history, "baseline", output_dir)
    plot_confusion_matrix(model, X_test, y_test, output_dir, "baseline")
    plot_misclassified_examples(model, X_test, y_test, output_dir, "baseline")
    plot_weight_visualization(model, output_dir, "baseline")
    plot_sample_predictions(model, X_test, y_test, output_dir, "baseline")
    plot_per_class_accuracy(model, X_test, y_test, output_dir, "baseline")
    
    return model, history

def experiment_hidden_size(X_train, y_train, X_test, y_test, hidden_sizes, output_dir='output'):
    print("\n=== Hidden Size Experiment ===")
    
    input_size = X_train.shape[1]
    output_size = 10
    epochs = 50
    batch_size = 32
    learning_rate = 0.01
    
    results = {}
    
    for hidden_size in hidden_sizes:
        print(f"\nTraining with hidden_size={hidden_size}...")
        model = NeuralNetwork(input_size, hidden_size, output_size)
        history = train_network(model, X_train, y_train, X_test, y_test, 
                              epochs, batch_size, learning_rate)
        results[hidden_size] = {'history': history, 'model': model}
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for hidden_size in hidden_sizes:
        plt.plot(results[hidden_size]['history']['test_losses'], label=f'Hidden={hidden_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Hidden Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for hidden_size in hidden_sizes:
        plt.plot(results[hidden_size]['history']['test_accuracies'], label=f'Hidden={hidden_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Hidden Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hidden_size_comparison.png', dpi=150)
    plt.close()
    print("Saved: hidden_size_comparison.png")
    
    # Plot final metrics comparison
    history_dict = {hs: results[hs]['history'] for hs in hidden_sizes}
    plot_loss_comparison_bar(history_dict, output_dir, 'hidden_size', 'Hidden Size')
    
    # Generate detailed plots for each hidden size
    for hidden_size in hidden_sizes:
        model = results[hidden_size]['model']
        exp_name = f"hidden_{hidden_size}"
        plot_confusion_matrix(model, X_test, y_test, output_dir, exp_name)
        plot_per_class_accuracy(model, X_test, y_test, output_dir, exp_name)
    
    return results

def experiment_learning_rate(X_train, y_train, X_test, y_test, learning_rates, output_dir='output'):
    print("\n=== Learning Rate Experiment ===")
    
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 10
    epochs = 50
    batch_size = 32
    
    results = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning_rate={lr}...")
        model = NeuralNetwork(input_size, hidden_size, output_size)
        history = train_network(model, X_train, y_train, X_test, y_test, 
                              epochs, batch_size, lr)
        results[lr] = {'history': history, 'model': model}
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for lr in learning_rates:
        plt.plot(results[lr]['history']['test_losses'], label=f'LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for lr in learning_rates:
        plt.plot(results[lr]['history']['test_accuracies'], label=f'LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_rate_comparison.png', dpi=150)
    plt.close()
    print("Saved: learning_rate_comparison.png")
    
    # Plot final metrics comparison
    history_dict = {lr: results[lr]['history'] for lr in learning_rates}
    plot_loss_comparison_bar(history_dict, output_dir, 'learning_rate', 'Learning Rate')
    
    # Generate detailed plots for each learning rate
    for lr in learning_rates:
        model = results[lr]['model']
        exp_name = f"lr_{lr}"
        plot_confusion_matrix(model, X_test, y_test, output_dir, exp_name)
        plot_per_class_accuracy(model, X_test, y_test, output_dir, exp_name)
    
    return results
