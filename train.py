import numpy as np
from utils import softmax, cross_entropy_loss, compute_accuracy, one_hot_encode

def train_network(model, X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
    n_samples = X_train.shape[0]
    num_classes = model.output_size
    
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Shuffle training data at the start of each epoch
        np.random.seed(epoch)
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_onehot[indices]
        
        # Mini-batch SGD
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_train_shuffled[i:batch_end]
            y_batch = y_train_shuffled[i:batch_end]
            
            # Forward pass
            logits = model.forward(X_batch)
            y_pred = softmax(logits)
            
            # Backward pass
            model.backward(y_pred, y_batch, learning_rate)
        
        # Evaluate on training set
        train_pred = model.predict(X_train)
        train_loss = cross_entropy_loss(train_pred, y_train_onehot)
        train_acc = compute_accuracy(train_pred, y_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate on test set
        test_pred = model.predict(X_test)
        test_loss = cross_entropy_loss(test_pred, y_test_onehot)
        test_acc = compute_accuracy(test_pred, y_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
