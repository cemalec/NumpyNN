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
        np.savez(filepath,
                 layers=[dict(input_size=layer.input_size,
                              output_size=layer.output_size,
                              weights=layer.weights,
                              biases=layer.biases,
                              activation_function=layer.activation_function.__class__.__name__,
                              last_input=layer.last_input,
                              last_z=layer.last_z) for layer in self.layers],
                 loss=self.loss.__class__.__name__)
    
    @classmethod
    def load(cls, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        load_layers = data['layers']
        load_loss = data['loss']
        layers = []
        for layer in load_layers:
            init_layer = DenseLayer(layer['input_size'],
                                    layer['output_size'],
                                    activation_function=getattr(__import__('DifferentiableFunction'), layer['activation_function'])())
            init_layer.weights = layer['weights']
            init_layer.biases = layer['biases']
            init_layer.last_input = layer['last_input']
            init_layer.last_z = layer['last_z']
            layers.append(init_layer)
        loss = getattr(__import__('DifferentiableFunction'), load_loss.item())()
        return cls(layers=layers, loss=loss)