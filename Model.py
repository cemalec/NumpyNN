from typing import List
import numpy as np
from Layer import (
    BatchNormLayer,
    CNNLayer,
    DenseLayer,
    EmbeddingLayer,
    FlattenLayer,
    LayerGradients,
    MaxPoolLayer,
    ReshapeLayer,
)
from DifferentiableFunction import CrossEntropyLoss, DifferentiableFunction, ReLU, Sigmoid, SoftMax
from Optimizer import Adam, Optimizer, RMSProp, SGD

LAYER_TYPES = {
    "Dense": DenseLayer,
    "DenseLayer": DenseLayer,
    "CNN": CNNLayer,
    "CNNLayer": CNNLayer,
    "Flatten": FlattenLayer,
    "FlattenLayer": FlattenLayer,
    "Reshape": ReshapeLayer,
    "ReshapeLayer": ReshapeLayer,
    "MaxPool": MaxPoolLayer,
    "MaxPoolLayer": MaxPoolLayer,
    "BatchNormLayer": BatchNormLayer,
    "EmbeddingLayer": EmbeddingLayer,
}
LOSS_TYPES = {"CrossEntropyLoss": CrossEntropyLoss}
OPTIMIZER_TYPES = {"SGD": SGD, "RMSProp": RMSProp, "Adam": Adam}


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

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, BatchNormLayer):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        upstream_gradient = self.loss.derivative(y_true, y_pred)
        last_layer_index = len(self.layers) - 1
        uses_fused_softmax_cross_entropy = (
            isinstance(self.loss, CrossEntropyLoss)
            and isinstance(self.layers[-1], DenseLayer)
            and isinstance(self.layers[-1].activation_function, SoftMax)
        )

        for layer_index in range(last_layer_index, -1, -1):
            layer = self.layers[layer_index]
            if layer_index == last_layer_index and uses_fused_softmax_cross_entropy:
                gradients = layer.backward(
                    upstream_gradient, apply_activation_derivative=False
                )
            else:
                gradients = layer.backward(upstream_gradient)

            if gradients.parameter_gradients:
                self.optimizer.step(layer, gradients.parameter_gradients)

            upstream_gradient = gradients.input_gradient

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)

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
        layers = []
        for layer_data in data["layers"]:
            layer_type = layer_data["type"]
            if layer_type not in LAYER_TYPES:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            layers.append(LAYER_TYPES[layer_type].from_dict(layer_data))

        loss_name = data["loss"]
        if loss_name not in LOSS_TYPES:
            raise ValueError(f"Unsupported loss type: {loss_name}")
        loss = LOSS_TYPES[loss_name]()

        optimizer_data = data["optimizer"]
        optimizer_name = optimizer_data["type"]
        if optimizer_name not in OPTIMIZER_TYPES:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
        optimizer = OPTIMIZER_TYPES[optimizer_name].from_dict(optimizer_data)
        return cls(layers=layers, loss=loss, optimizer=optimizer)

    def save(self, filepath: str):
        """Save model to npz file with proper serialization."""
        model_dict = self.to_dict()
        save_dict = {"model_config": model_dict}

        # Save all learnable parameters for each layer
        for i, layer in enumerate(self.layers):
            for param_name, parameter in layer.parameters().items():
                save_dict[f"layer_{i}_{param_name}"] = parameter

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
            parameter_prefix = f"layer_{i}_"
            has_saved_parameters = any(
                key.startswith(parameter_prefix) for key in data.files
            )
            if has_saved_parameters and not layer.parameters():
                layer.initialize_weights()
                layer.weights_initialized = True

            for param_name in layer.parameters():
                key = f"{parameter_prefix}{param_name}"
                if key in data:
                    setattr(layer, param_name, data[key])
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
