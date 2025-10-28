from Model import Model
from Layer import DenseLayer
from Dataset import *
from Optimizer import *
from metrics import *
from DifferentiableFunction import DifferentiableFunction,SoftMax,ReLU,CrossEntropyLoss

import numpy as np
from typing import List
import argparse
import logging

INPUT_SIZE = 28*28
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 10

#basic_model = Model.load('models/adam_model.npz')
def training_loop(model: Model, 
                  dataset: Dataset,
                  batch_size: int,
                  epochs: int, 
                  learning_rate: float):
    for epoch in range(epochs):
        i = 0
        batch_accuracies = []
        batch_losses = []
        for x_train, y_train in dataset.get_batch(batch_size):
            # Forward pass
            y_pred = model.forward(x_train)
            
            # Compute loss
            loss = model.compute_loss(y_train, y_pred)
            batch_losses.append(loss)
            
            # Backward pass
            model.backward(y_train, y_pred, learning_rate)
            batch_acc = accuracy(y_train, y_pred)
            batch_accuracies.append(batch_acc)
            i+=1
        avg_acc = np.mean(batch_accuracies)
        avg_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a simple neural network on MNIST")
    argparser.add_argument('--epochs', 
                           type=int, 
                           default=20, 
                           help='Number of epochs to train')
    argparser.add_argument('--batch_size', 
                           type=int, 
                           default=32, 
                           help='Batch size for training')
    argparser.add_argument('--learning_rate', 
                           type=float, 
                           default=1e-3, 
                           help='Learning rate for optimizer')
    argparser.add_argument('--log_level',
                           type=str,
                           default='INFO')
    
    args = argparser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    basic_model = Model(
        layers=[DenseLayer(input_size=INPUT_SIZE,
                           output_size=HIDDEN_SIZE1,
                           activation_function=ReLU()), 
                DenseLayer(input_size=HIDDEN_SIZE1,
                           output_size=HIDDEN_SIZE2,
                           activation_function=ReLU()),
                DenseLayer(input_size=HIDDEN_SIZE2,
                           output_size=OUTPUT_SIZE,
                           activation_function=SoftMax())],
        loss=CrossEntropyLoss(),
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    train_dataset = MNISTDataset(split='train')
    test_dataset = MNISTDataset(split='test')
    
    X_train, y_train = train_dataset.X, train_dataset.y
    X_test, y_test = test_dataset.X, test_dataset.y
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
    
    training_loop(model=basic_model,
                  dataset=train_dataset,
                  batch_size=batch_size,
                  epochs=epochs,
                  learning_rate=learning_rate)
    basic_model.save('models/bigger_model.npz')
    
    # Evaluate on training set
    y_train_pred = basic_model.predict(X_train)
    train_loss = basic_model.compute_loss(y_train, y_train_pred)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {accuracy(y_train, y_train_pred):.4f}")
    
    # Evaluate on test set
    y_test_pred = basic_model.predict(X_test)
    test_loss = basic_model.compute_loss(y_test, y_test_pred)
    test_accuracy = accuracy(y_test, y_test_pred)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")