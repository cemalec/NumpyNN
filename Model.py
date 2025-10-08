from typing import List
import numpy as np
from Layer import DenseLayer
from DifferentiableFunction import DifferentiableFunction,SoftMax

class Model:
    layers: List[DenseLayer]
    loss: DifferentiableFunction

    def __init__(self, layers: List[DenseLayer], loss: DifferentiableFunction):
        self.layers = layers
        self.loss = loss
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        loss_grad = self.loss.derivative(y_true,y_pred)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(self.loss.function(y_true, y_pred))
    
    def save(self, filepath: str):
        np.savez(filepath, layers=self.layers, loss=self.loss)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.layers = data['layers'].tolist()
        self.loss = data['loss'].item()