from typing import List
import numpy as np
from Layer import DenseLayer
from DifferentiableFunction import DifferentiableFunction
from Optimizer import Optimizer


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

            # Check if any gradient exists for layer parameters (excluding 'inputs')
            has_learnable_params = any(
                grad_dict.get(param) is not None
                for param in grad_dict.keys()
                if param != "inputs"
            )

            if has_learnable_params:
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
        layers = [
            getattr(__import__("Layer"), layer_data["type"]).from_dict(layer_data)
            for layer_data in data["layers"]
        ]
        loss = getattr(__import__("DifferentiableFunction"), data["loss"])()
        optimizer = getattr(
            __import__("Optimizer"), data["optimizer"]["type"]
        ).from_dict(data["optimizer"])
        return cls(layers=layers, loss=loss, optimizer=optimizer)

    def save(self, filepath: str):
        """Save model to npz file with proper serialization."""
        model_dict = self.to_dict()
        save_dict = {"model_config": model_dict}

        # Save all learnable parameters for each layer
        for i, layer in enumerate(self.layers):
            for param_name in ["weights", "biases", "gamma", "beta"]:
                if hasattr(layer, param_name):
                    param = getattr(layer, param_name)
                    if param is not None:
                        save_dict[f"layer_{i}_{param_name}"] = param

        np.savez(filepath, **save_dict)

        if hasattr(self.optimizer, "save_state"):
            opt_filepath = filepath.replace(".npz", "_optimizer.npz")
            self.optimizer.save_state(opt_filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load model from npz file."""
        data = np.load(filepath, allow_pickle=True)
        model_config = data["model_config"].item()

        model = cls.from_dict(model_config)

        # Restore all learnable parameters
        for i, layer in enumerate(model.layers):
            for param_name in ["weights", "biases", "gamma", "beta"]:
                key = f"layer_{i}_{param_name}"
                if key in data and hasattr(layer, param_name):
                    setattr(layer, param_name, data[key])
                    if param_name in ["weights", "biases"]:
                        layer.weights_initialized = True

        # Load optimizer state if available
        opt_filepath = filepath.replace(".npz", "_optimizer.npz")
        if hasattr(model.optimizer.__class__, "load_state"):
            try:
                model.optimizer = model.optimizer.__class__.load_state(
                    opt_filepath, learning_rate=model.optimizer.learning_rate
                )
            except FileNotFoundError:
                pass

        return model
