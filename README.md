# Neural Network from Scratch - Homework 1

A complete implementation of a feed-forward neural network from scratch using only NumPy. This project trains on the MNIST handwritten digit dataset and performs comprehensive experiments with different hyperparameters.

## Quick Start

### 1. Setup Virtual Environment
```powershell
cd "d:\Projects\Golf swing analysis\639-Hw1"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

### 2. Download MNIST Dataset
Download these 4 files from http://yann.lecun.com/exdb/mnist/:
- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Extract the `.gz` files and place the extracted files in the `639-Hw1` folder.

### 3. Run All Experiments
```powershell
python hw1.py
```

## Project Structure

```
639-Hw1/
├── hw1.py                      # Main script - run this!
├── neural_network.py           # Neural network class (forward/backward pass)
├── data_loader.py              # MNIST data loading and preprocessing
├── utils.py                    # Activation functions, loss, accuracy
├── train.py                    # Training loop with mini-batch SGD
├── experiments.py              # Experiment configurations
├── visualization.py            # Plotting and visualization functions
├── README.md                   # This file
├── train-images.idx3-ubyte     # MNIST training images (download)
├── train-labels.idx1-ubyte     # MNIST training labels (download)
├── t10k-images.idx3-ubyte      # MNIST test images (download)
├── t10k-labels.idx1-ubyte      # MNIST test labels (download)
└── output_YYYYMMDD_HHMMSS/     # Generated output folder with all results
```

## File Descriptions

| File | Purpose |
|------|---------|
| **hw1.py** | Main entry point - runs all experiments and creates output folder |
| **neural_network.py** | `NeuralNetwork` class with forward/backward propagation |
| **data_loader.py** | Loads and preprocesses MNIST dataset |
| **utils.py** | Helper functions (ReLU, softmax, cross-entropy, accuracy) |
| **train.py** | Training loop with mini-batch SGD |
| **experiments.py** | Three main experiments (baseline, hidden size, learning rate) |
| **visualization.py** | Generates confusion matrices, weight visualizations, etc. |

## Experiments

### Experiment 1: Baseline
- **Hidden Units**: 128
- **Learning Rate**: 0.01
- **Epochs**: 50
- **Batch Size**: 32

### Experiment 2: Hidden Size Comparison
Tests different hidden layer sizes: **64, 128, 256**

### Experiment 3: Learning Rate Comparison
Tests different learning rates: **1.0, 0.1, 0.01, 0.001**

## Output Visualizations

All outputs are saved to a timestamped folder `output_YYYYMMDD_HHMMSS/` containing **23+ plots**:

### Baseline Experiment (6 plots)
- `baseline_learning_curves.png` - Training/test loss and accuracy curves
- `baseline_confusion_matrix.png` - 10x10 confusion matrix
- `baseline_misclassified.png` - 20 misclassified examples
- `baseline_weights.png` - First layer weight visualizations
- `baseline_samples.png` - 20 random predictions with confidence scores
- `baseline_per_class_accuracy.png` - Accuracy for each digit (0-9)

### Hidden Size Comparison (8+ plots)
- `hidden_size_comparison.png` - Loss/accuracy curves overlay
- `hidden_size_final_metrics.png` - Final metrics bar chart
- `hidden_64_confusion_matrix.png` - Confusion matrix (64 units)
- `hidden_128_confusion_matrix.png` - Confusion matrix (128 units)
- `hidden_256_confusion_matrix.png` - Confusion matrix (256 units)
- `hidden_*_per_class_accuracy.png` - Per-class accuracy for each size

### Learning Rate Comparison (8+ plots)
- `learning_rate_comparison.png` - Loss/accuracy curves overlay
- `learning_rate_final_metrics.png` - Final metrics bar chart
- `lr_1.0_confusion_matrix.png` - Confusion matrix (lr=1.0)
- `lr_0.1_confusion_matrix.png` - Confusion matrix (lr=0.1)
- `lr_0.01_confusion_matrix.png` - Confusion matrix (lr=0.01)
- `lr_0.001_confusion_matrix.png` - Confusion matrix (lr=0.001)
- `lr_*_per_class_accuracy.png` - Per-class accuracy for each rate

## Implementation Details

### Architecture
- **Input Layer**: 784 units (28x28 pixels) + 1 bias
- **Hidden Layer**: Configurable size + 1 bias
- **Output Layer**: 10 units (digits 0-9)

### Technical Specifications
- **Activation**: ReLU (hidden layer), Linear (output layer)
- **Loss Function**: Cross-entropy with numerically stable softmax
- **Optimizer**: Mini-batch SGD (batch size = 32)
- **Weight Initialization**: Uniform(-0.01, 0.01) with np.random.seed(0)
- **Data Processing**: Normalized to [0, 1], shuffled each epoch

### Key Features
✅ Fully vectorized operations (no loops over samples)  
✅ Numerically stable softmax (subtracts max logit)  
✅ Bias units in input and hidden layers  
✅ Consistent random seeding for reproducibility  
✅ Comprehensive visualization suite  

## Requirements

```
numpy
matplotlib
```

Install with:
```powershell
pip install numpy matplotlib
```

**Note**: No deep learning frameworks (PyTorch, TensorFlow) are used - this is a from-scratch implementation!

## Troubleshooting

### MNIST Files Not Found
Make sure you've extracted the `.gz` files and placed all 4 `.idx3-ubyte` and `.idx1-ubyte` files in the same directory as `hw1.py`.

### Execution Policy Error (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors
Make sure your virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

## Assignment Compliance

✅ Feed-forward neural network from scratch  
✅ One hidden layer with configurable units  
✅ Bias units in input and hidden layers  
✅ ReLU activation (hidden), linear output  
✅ Numerically stable softmax  
✅ np.random.seed(0) set before initialization  
✅ Weights from U(-0.01, 0.01)  
✅ Mini-batch SGD (batch size 32)  
✅ Data shuffled each epoch  
✅ Only permitted libraries used (numpy, matplotlib, struct, os, datetime)  
✅ Vectorized matrix operations  
✅ Well-structured, commented code  

## Contact

For questions about this implementation, refer to the course staff or assignment documentation.
