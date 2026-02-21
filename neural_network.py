import numpy as np
from utils import relu, relu_derivative, softmax

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with U(-0.01, 0.01)
        np.random.seed(0)
        
        # W1: (input_size + 1) x hidden_size (includes bias)
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size + 1, hidden_size))
        
        # W2: (hidden_size + 1) x output_size (includes bias)
        self.W2 = np.random.uniform(-0.01, 0.01, (hidden_size + 1, output_size))
        
        # Cache for backward pass
        self.cache = {}
    
    def add_bias(self, X):
        # Add bias column of ones
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    
    def forward(self, X):
        # Input layer with bias
        X_with_bias = self.add_bias(X)
        
        # Hidden layer
        z1 = np.dot(X_with_bias, self.W1)
        a1 = relu(z1)
        
        # Add bias to hidden layer
        a1_with_bias = self.add_bias(a1)
        
        # Output layer (linear)
        z2 = np.dot(a1_with_bias, self.W2)
        
        # Cache for backward pass
        self.cache = {
            'X': X,
            'X_with_bias': X_with_bias,
            'z1': z1,
            'a1': a1,
            'a1_with_bias': a1_with_bias,
            'z2': z2
        }
        
        return z2
    
    def backward(self, y_pred, y_true, learning_rate):
        m = y_true.shape[0]
        
        # Output layer gradient
        dz2 = y_pred - y_true
        dW2 = np.dot(self.cache['a1_with_bias'].T, dz2) / m
        
        # Hidden layer gradient
        da1 = np.dot(dz2, self.W2[:-1, :].T)  # Exclude bias weights
        dz1 = da1 * relu_derivative(self.cache['z1'])
        dW1 = np.dot(self.cache['X_with_bias'].T, dz1) / m
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
    
    def predict(self, X):
        logits = self.forward(X)
        probabilities = softmax(logits)
        return probabilities
