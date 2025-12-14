from typing import List
import numpy as np
from Layer import DenseLayer
from DifferentiableFunction import DifferentiableFunction, SoftMax
from Optimizer import *


class Model:
    layers: List[DenseLayer]
    loss: DifferentiableFunction
    optimizer: Optimizer

    def __init__(
        self,
        layers: List[DenseLayer],
        loss: DifferentiableFunction,
        optimizer: Optimizer,
    ):
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

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        loss_grad = self.loss.derivative(y_true, y_pred)
        grad_dict = {"inputs": loss_grad}
        for layer in reversed(self.layers):
            grad_dict = layer.backward(grad_dict["inputs"])
            # Only call optimizer step if layer has learnable parameters
            if grad_dict["weights"] is not None and grad_dict["biases"] is not None:
                self.optimizer.step(layer, grad_dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(self.loss.function(y_true, y_pred))

    def to_dict(self):
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "loss": self.loss.__class__.__name__,
            "optimizer": self.optimizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        layers = [getattr(__import__("Layer"), layer_data["type"]).from_dict(layer_data) for layer_data in data["layers"]]
        loss = getattr(__import__("DifferentiableFunction"), data["loss"])()
        optimizer = getattr(
            __import__("Optimizer"), data["optimizer"]["type"]
        ).from_dict(data["optimizer"])
        return cls(layers=layers, loss=loss, optimizer=optimizer)

    def save(self, filepath: str):
        """Save model to npz file with proper serialization."""
        model_dict = self.to_dict()
        
        save_dict = {'model_config': model_dict}
        
        # Save layer weights and biases
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and layer.weights is not None:
                save_dict[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'biases') and layer.biases is not None:
                save_dict[f'layer_{i}_biases'] = layer.biases
        
        np.savez(filepath, **save_dict)
        
        # Save optimizer state separately if it has state
        if hasattr(self.optimizer, 'save_state'):
            opt_filepath = filepath.replace('.npz', '_optimizer.npz')
            self.optimizer.save_state(opt_filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load model from npz file."""
        data = np.load(filepath, allow_pickle=True)
        model_config = data['model_config'].item()
        
        model = cls.from_dict(model_config)
        
        # Restore weights and biases
        for i, layer in enumerate(model.layers):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_biases'
            
            if weight_key in data:
                layer.weights = data[weight_key]
                layer.weights_initialized = True
            if bias_key in data:
                layer.biases = data[bias_key]
        
        # Load optimizer state if available
        opt_filepath = filepath.replace('.npz', '_optimizer.npz')
        if hasattr(model.optimizer.__class__, 'load_state'):
            try:
                model.optimizer = model.optimizer.__class__.load_state(
                    opt_filepath, 
                    learning_rate=model.optimizer.learning_rate
                )
            except FileNotFoundError:
                pass  # Continue with fresh optimizer state
    
        return model
