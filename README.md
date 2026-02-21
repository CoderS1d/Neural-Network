# Neural Network from Scratch - Homework 1

## Project Structure

```
639-Hw1/
├── hw1.py                  # Main script to run all experiments
├── neural_network.py       # Neural network implementation
├── data_loader.py          # MNIST data loading functions
├── utils.py                # Helper functions (ReLU, softmax, etc.)
├── train.py                # Training logic with mini-batch SGD
├── experiments.py          # Experiment functions and plotting
├── visualization.py        # Additional visualization functions
├── README.md               # This file
└── output_YYYYMMDD_HHMMSS/ # Generated output folder with all plots
```

## File Descriptions

- **hw1.py**: Main entry point that runs all experiments and generates plots
- **neural_network.py**: Contains the `NeuralNetwork` class with forward and backward pass
- **data_loader.py**: Functions to load and preprocess MNIST dataset
- **utils.py**: Utility functions for activation, loss, and accuracy calculations
- **train.py**: Training loop implementation with mini-batch SGD
- **experiments.py**: Different experiment functions that vary hyperparameters
- **visualization.py**: Functions for generating confusion matrices, weight visualizations, etc.

## Requirements

- numpy
- matplotlib
- MNIST dataset files (place in same directory as hw1.py):
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

## Usage

Run all experiments:
```bash
python hw1.py
```

This will:
1. Load MNIST dataset
2. Run baseline experiment (128 hidden units, lr=0.01)
3. Run hidden size comparison experiment (64, 128, 256)
4. Run learning rate comparison experiment (0.001, 0.01, 0.1)
5. Generate plots for all experiments

## Output

All outputs are saved to a timestamped folder `output_YYYYMMDD_HHMMSS/` containing:

### Baseline Experiment Plots:
- `baseline_learning_curves.png` - Training/test loss and accuracy over epochs
- `baseline_confusion_matrix.png` - Confusion matrix showing predictions vs true labels
- `baseline_misclassified.png` - Grid of misclassified examples
- `baseline_weights.png` - Visualization of first layer weights
- `baseline_samples.png` - Random sample predictions with confidence scores
- `baseline_per_class_accuracy.png` - Accuracy breakdown by digit class

### Hidden Size Comparison Plots:
- `hidden_size_comparison.png` - Loss and accuracy curves for different hidden sizes
- `hidden_size_final_metrics.png` - Bar chart comparing final train/test metrics
- `hidden_64_confusion_matrix.png` - Confusion matrix for 64 hidden units
- `hidden_128_confusion_matrix.png` - Confusion matrix for 128 hidden units
- `hidden_256_confusion_matrix.png` - Confusion matrix for 256 hidden units
- `hidden_*_per_class_accuracy.png` - Per-class accuracy for each configuration

### Learning Rate Comparison Plots:
- `learning_rate_comparison.png` - Loss and accuracy curves for different learning rates
- `learning_rate_final_metrics.png` - Bar chart comparing final train/test metrics
- `lr_0.001_confusion_matrix.png` - Confusion matrix for lr=0.001
- `lr_0.01_confusion_matrix.png` - Confusion matrix for lr=0.01
- `lr_0.1_confusion_matrix.png` - Confusion matrix for lr=0.1
- `lr_*_per_class_accuracy.png` - Per-class accuracy for each configuration

## Implementation Details

- **Architecture**: One hidden layer with configurable size
- **Activation**: ReLU for hidden layer, linear for output
- **Loss**: Cross-entropy with softmax (numerically stable)
- **Optimizer**: Mini-batch SGD (batch size = 32)
- **Weight Initialization**: Uniform(-0.01, 0.01) with seed=0
- **Bias**: Included in both input and hidden layers
