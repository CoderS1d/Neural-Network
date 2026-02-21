import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(logits):
    # Numerical stability: subtract max logit
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / n
    return loss

def compute_accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        labels = np.argmax(y_true, axis=1)
    else:
        labels = y_true
    return np.mean(predictions == labels)

def one_hot_encode(y, num_classes):
    n = y.shape[0]
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot
