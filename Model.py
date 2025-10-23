from typing import List
import numpy as np
from Layer import DenseLayer
from DifferentiableFunction import DifferentiableFunction,SoftMax
from Optimizer import *

class Model:
    layers: List[DenseLayer]
    loss: DifferentiableFunction
    optimizer: Optimizer

    def __init__(self, layers: List[DenseLayer], loss: DifferentiableFunction, optimizer: Optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        for layer in self.layers:
            if layer.name is None:
                layer.name = f"Layer_{self.layers.index(layer)}"

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        loss_grad = self.loss.derivative(y_true,y_pred)
        grad_dict = {'inputs': loss_grad}
        for layer in reversed(self.layers):
            grad_dict = layer.backward(grad_dict['inputs'])
            self.optimizer.step(layer,grad_dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(self.loss.function(y_true, y_pred))
    
    def to_dict(self):
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'loss': self.loss.__class__.__name__,
            'optimizer': self.optimizer.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        layers = [DenseLayer.from_dict(layer_data) for layer_data in data['layers']]
        loss = getattr(__import__('DifferentiableFunction'), data['loss'])()
        optimizer = getattr(__import__('Optimizer'), data['optimizer']['type']).from_dict(data['optimizer'])
        return cls(layers=layers, loss=loss, optimizer=optimizer)
    
    def save(self, filepath: str):
        np.savez(filepath,
                 layers=[dict(name=layer.name,
                              input_size=layer.input_size,
                              output_size=layer.output_size,
                              weights=layer.weights,
                              biases=layer.biases,
                              activation_function=layer.activation_function.__class__.__name__,
                              last_input=layer.last_input,
                              last_z=layer.last_z) for layer in self.layers],
                 loss=self.loss.__class__.__name__,
                 optimizer_name=self.optimizer.__class__.__name__,
                 optimizer=self.optimizer.to_dict(),
                 allows_pickle=True)
    
    @classmethod
    def load(cls, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        load_layers = data['layers']
        load_loss = data['loss']
        layers = []
        for layer in load_layers:
            init_layer = DenseLayer(layer['input_size'],
                                    layer['output_size'],
                                    activation_function=getattr(__import__('DifferentiableFunction'), layer['activation_function'])(),
                                    name=layer['name'])
            init_layer.weights = layer['weights']
            init_layer.biases = layer['biases']
            init_layer.last_input = layer['last_input']
            init_layer.last_z = layer['last_z']
            layers.append(init_layer)
        loss = getattr(__import__('DifferentiableFunction'), load_loss.item())()
        optimizer = getattr(__import__('Optimizer'), data['optimizer_name'].item()).from_dict(data['optimizer'].item()) 
        return cls(layers=layers, loss=loss,optimizer=optimizer)