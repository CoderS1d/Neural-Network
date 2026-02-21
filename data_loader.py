import numpy as np
import struct

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist_data(train_images_path, train_labels_path, test_images_path, test_labels_path):
    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)
    
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Normalize to [0,1]
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0
    
    return X_train, y_train, X_test, y_test
