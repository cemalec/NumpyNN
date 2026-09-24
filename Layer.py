from typing import Tuple, List
import numpy as np
from DifferentiableFunction import DifferentiableFunction, GeLU, ReLU, Sigmoid, SoftMax
from typing import Dict
from abc import abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

ACTIVATION_TYPES = {
    activation.__name__: activation for activation in (GeLU, ReLU, Sigmoid, SoftMax)
}


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class LayerGradients:
    """Gradients produced by one layer's backward pass."""

    input_gradient: np.ndarray | None
    parameter_gradients: Dict[str, np.ndarray]


class Layer:
    def __init__(self):
        self.name = None
        self.type = "Layer"
        self.weights = None
        self.biases = None
        self.last_input = None
        self.last_z = None
        self.weights_initialized = False

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights_initialized is False:
            self.initialize_weights()
            self.weights_initialized = True

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        pass

    def parameters(self) -> Dict[str, np.ndarray]:
        """Return this layer's trainable arrays by name."""
        return {
            name: parameter
            for name, parameter in {
                "weights": self.weights,
                "biases": self.biases,
            }.items()
            if parameter is not None
        }

    def set_parameter(self, name: str, value: np.ndarray) -> None:
        """Replace one trainable array returned by ``parameters()``."""
        setattr(self, name, value)

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> "Layer":
        pass


class DenseLayer(Layer):
    """
    A fully connected neural network layer.

     Parameters:
        input_size (int): The number of input features.
        output_size (int): The number of output features (neurons).
        activation_function (DifferentiableFunction): The activation function to apply.
        name (str, optional): Name of the layer. Defaults to None.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: DifferentiableFunction,
        name: str = None,
    ):
        super().__init__()
        _require_positive_int(input_size, "input_size")
        _require_positive_int(output_size, "output_size")
        self.name = name
        self.type = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def initialize_weights(self):
        """
        initialize weights with He initialization.
        """
        self.weights = np.random.randn(self.input_size, self.output_size) * (
            np.sqrt(2.0 / self.input_size)
        )
        self.biases = np.zeros(self.output_size)
        logger.info(f"Weights and biases initialized for layer {self.name}")

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer, supporting batched input.
        Parameters:
            input_data (np.ndarray): Input data to the layer. Shape: (batch_size, input_size)

        Returns:
            np.ndarray: Output after applying weights, biases, and activation function. Shape: (batch_size, output_size)
        """
        if input_data.ndim < 2:
            raise ValueError("DenseLayer expects at least 2D input")
        if input_data.shape[-1] != self.input_size:
            raise ValueError("DenseLayer input feature dimension does not match input_size")

        super().forward(input_data)
        self.last_input = input_data
        self.last_z = input_data @ self.weights + self.biases
        logger.debug(
            f"Forward pass in layer {self.name}: input shape {input_data.shape}, z shape {self.last_z.shape}"
        )
        return self.activation_function.function(self.last_z)

    def backward(
        self, dL_da: np.ndarray, apply_activation_derivative: bool = True
    ) -> LayerGradients:
        """
        Performs the backward pass through the layer, updating weights and biases.
        Parameters:
            dL_da (np.ndarray): Gradient of the loss with respect to the layer's output. Shape: (batch_size, output_size)
            learning_rate (float): Learning rate for weight updates.
        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input. Shape: (batch_size, input_size)
        """
        # The gradient of the activation function with respect to the scores (last_z)
        da_dz = (
            self.activation_function.derivative(self.last_z)
            if apply_activation_derivative
            else 1
        )
        # The gradient of the loss with respect to the scores
        dL_dz = dL_da * da_dz  # (batch_size, output_size)
        # Flatten batch-like dimensions while preserving the final feature axis.
        input_matrix = self.last_input.reshape(-1, self.input_size)
        score_gradient_matrix = dL_dz.reshape(-1, self.output_size)
        weight_gradient = input_matrix.T @ score_gradient_matrix
        bias_gradient = np.sum(score_gradient_matrix, axis=0)
        input_gradient = dL_dz @ self.weights.T

        logger.debug(
            f"Backward pass in layer {self.name}: output_gradient shape {dL_da.shape}, input_gradient shape {input_gradient.shape}"
        )
        return LayerGradients(
            input_gradient=input_gradient,
            parameter_gradients={
                "weights": weight_gradient,
                "biases": bias_gradient,
            },
        )

        # Update weights and biases will be handled by the optimizer in Model.py

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation_function": self.activation_function.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DenseLayer":
        activation_name = data["activation_function"]
        if activation_name not in ACTIVATION_TYPES:
            raise ValueError(f"Unsupported activation type: {activation_name}")
        activation_function = ACTIVATION_TYPES[activation_name]()
        layer = cls(
            input_size=data["input_size"],
            output_size=data["output_size"],
            activation_function=activation_function,
            name=data["name"],
        )
        return layer


class CNNLayer(Layer):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        output_size: Tuple[int, int, int],
        kernel_size: int | List[int],
        num_filters: int,
        padding: int = 0,
        stride: int = 1,
        activation_function: DifferentiableFunction = None,
        name: str = None,
    ):
        super().__init__()
        if len(input_size) != 3 or any(dimension <= 0 for dimension in input_size):
            raise ValueError("CNNLayer input_size must contain three positive dimensions")
        if len(output_size) != 3 or any(dimension <= 0 for dimension in output_size):
            raise ValueError("CNNLayer output_size must contain three positive dimensions")
        _require_positive_int(kernel_size, "kernel_size")
        _require_positive_int(num_filters, "num_filters")
        _require_positive_int(stride, "stride")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        self.name = name
        self.type = "CNN"
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride
        self.activation_function = activation_function

    def initialize_weights(self):
        # Need in_channels - extract from input_size or add as parameter
        in_channels = (
            self.input_size[0] if isinstance(self.input_size, (list, tuple)) else 1
        )
        self.weights = (
            np.random.randn(
                self.num_filters, in_channels, self.kernel_size, self.kernel_size
            )
            * 0.01
        )
        self.biases = np.zeros(self.num_filters)
        logger.info(f"Weights and biases initialized for CNN layer {self.name}")

    def pad_input(self, input_data: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            return np.pad(
                input_data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        return input_data

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim != 4:
            raise ValueError("CNNLayer expects 4D input")
        if input_data.shape[1] != self.input_size[0]:
            raise ValueError("CNNLayer input channel dimension does not match input_size")

        super().forward(input_data)
        input_data = self.pad_input(input_data)

        batch_size, in_channels, height, width = input_data.shape
        if height < self.kernel_size or width < self.kernel_size:
            raise ValueError("CNNLayer kernel_size exceeds input spatial dimensions")
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        if (self.num_filters, output_height, output_width) != tuple(self.output_size):
            raise ValueError("CNNLayer output_size does not match its configuration")

        # Create strided view of input for all windows at once
        # Shape: (batch, out_h, out_w, in_channels, filter_h, filter_w)
        shape = (
            batch_size,
            output_height,
            output_width,
            in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        strides = (
            input_data.strides[0],  # batch stride
            input_data.strides[2] * self.stride,  # output height stride
            input_data.strides[3] * self.stride,  # output width stride
            input_data.strides[1],  # channel stride
            input_data.strides[2],  # filter height stride
            input_data.strides[3],  # filter width stride
        )

        windows = np.lib.stride_tricks.as_strided(
            input_data, shape=shape, strides=strides
        )

        # Convolve: (batch, out_h, out_w, in_ch, fh, fw) with (num_filters, in_ch, fh, fw)
        # Result: (batch, out_h, out_w, num_filters)
        output = np.einsum(
            "bhwcij,fcij->bhwf",
            windows,
            self.weights.reshape(
                self.num_filters, in_channels, self.kernel_size, self.kernel_size
            ),
        )

        # Add biases and transpose to (batch, num_filters, out_h, out_w)
        output = np.transpose(output + self.biases, (0, 3, 1, 2))

        self.last_input = input_data
        self.last_z = output
        if self.activation_function is None:
            return output
        return self.activation_function.function(output)

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        """
        Vectorized backward pass for CNN layer.

        Parameters:
            output_gradient: Shape (batch, num_filters, out_h, out_w)

        Returns:
            Dictionary with 'inputs', 'weights', and 'biases' gradients
        """
        batch_size, num_filters, output_height, output_width = output_gradient.shape
        _, in_channels, padded_height, padded_width = self.last_input.shape

        if self.activation_function is not None:
            output_gradient = output_gradient * self.activation_function.derivative(
                self.last_z
            )

        # Bias gradient: sum over batch, height, and width
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3))

        # Prepare output gradient: (batch, num_filters, out_h, out_w) -> (batch, out_h, out_w, num_filters)
        dL_dout = np.transpose(output_gradient, (0, 2, 3, 1))

        # Weight gradient using strided windows
        # Create windows from last_input: (batch, out_h, out_w, in_ch, fh, fw)
        shape = (
            batch_size,
            output_height,
            output_width,
            in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        strides = (
            self.last_input.strides[0],
            self.last_input.strides[2] * self.stride,
            self.last_input.strides[3] * self.stride,
            self.last_input.strides[1],
            self.last_input.strides[2],
            self.last_input.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(
            self.last_input, shape=shape, strides=strides
        )

        # Weight gradient: (num_filters, in_ch, fh, fw)
        # dL_dout: (batch, out_h, out_w, num_filters)
        # windows: (batch, out_h, out_w, in_ch, fh, fw)
        weight_gradient = np.einsum("bhwf,bhwcij->fcij", dL_dout, windows)

        # Input gradient - need to do "full" convolution
        # Rotate filters 180 degrees for convolution
        rotated_weights = np.flip(self.weights, axis=(2, 3))

        # Pad output gradient for full convolution
        pad_h = self.kernel_size - 1
        pad_w = self.kernel_size - 1
        dL_dout_padded = np.pad(
            output_gradient,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )

        # Create strided view for backward pass
        # Shape: (batch, in_ch, padded_h, padded_w, num_filters, fh, fw)
        input_height = padded_height
        input_width = padded_width
        out_shape = (
            batch_size,
            input_height,
            input_width,
            num_filters,
            self.kernel_size,
            self.kernel_size,
        )
        out_strides = (
            dL_dout_padded.strides[0],
            dL_dout_padded.strides[2] * self.stride,
            dL_dout_padded.strides[3] * self.stride,
            dL_dout_padded.strides[1],
            dL_dout_padded.strides[2],
            dL_dout_padded.strides[3],
        )
        grad_windows = np.lib.stride_tricks.as_strided(
            dL_dout_padded, shape=out_shape, strides=out_strides
        )

        # Input gradient: (batch, in_ch, h, w)
        # grad_windows: (batch, h, w, num_filters, fh, fw)
        # rotated_weights: (num_filters, in_ch, fh, fw)
        input_gradient = np.einsum("bhwfij,fcij->bchw", grad_windows, rotated_weights)

        # Remove padding from input gradient if padding was applied
        if self.padding > 0:
            input_gradient = input_gradient[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        return LayerGradients(
            input_gradient=input_gradient,
            parameter_gradients={
                "weights": weight_gradient,
                "biases": bias_gradient,
            },
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "kernel_size": self.kernel_size,
            "num_filters": self.num_filters,
            "padding": self.padding,
            "stride": self.stride,
            "activation_function": (
                self.activation_function.__class__.__name__
                if self.activation_function is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CNNLayer":
        activation_name = data.get("activation_function")
        if activation_name is not None and activation_name not in ACTIVATION_TYPES:
            raise ValueError(f"Unsupported activation type: {activation_name}")
        activation_function = ACTIVATION_TYPES[activation_name]() if activation_name else None
        layer = cls(
            input_size=data["input_size"],
            output_size=data["output_size"],
            kernel_size=data["kernel_size"],
            num_filters=data["num_filters"],
            padding=data.get("padding", 0),
            stride=data.get("stride", 1),
            activation_function=activation_function,
            name=data.get("name"),
        )
        return layer


class FlattenLayer(Layer):
    """
    Flattens multi-dimensional input to 2D (batch_size, flattened_features).
    Useful for transitioning from CNN layers to Dense layers.
    """

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name
        self.type = "Flatten"
        self.input_shape = None

    def initialize_weights(self):
        """Flatten layer has no weights to initialize."""
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Flatten input from (batch, channels, height, width) to (batch, channels*height*width).

        Parameters:
            input_data: Shape (batch, ...) - any shape with batch as first dimension

        Returns:
            Flattened array of shape (batch, features)
        """
        super().forward(input_data)
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]

        logger.debug(f"Flatten layer {self.name}: input shape {input_data.shape}")

        # Flatten all dimensions except batch
        output = input_data.reshape(batch_size, -1)

        logger.debug(f"Flatten layer {self.name}: output shape {output.shape}")
        return output

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        """
        Reshape gradient back to original input shape.

        Parameters:
            output_gradient: Shape (batch, flattened_features)

        Returns:
            Dictionary with 'inputs' reshaped to original input shape
        """
        # Reshape back to the input shape
        input_gradient = output_gradient.reshape(self.input_shape)

        logger.debug(
            f"Flatten layer {self.name} backward: output_gradient shape {output_gradient.shape}, "
            f"input_gradient shape {input_gradient.shape}"
        )

        return LayerGradients(input_gradient=input_gradient, parameter_gradients={})

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FlattenLayer":
        return cls(name=data.get("name"))


class ReshapeLayer(Layer):
    """
    Reshapes input to a specified shape.
    Useful for converting between flattened and multi-dimensional formats.
    """

    def __init__(self, output_shape: Tuple[int, ...], name: str = None):
        super().__init__()
        if not output_shape or any(dimension <= 0 for dimension in output_shape):
            raise ValueError("output_shape must contain positive dimensions")
        self.name = name
        self.type = "Reshape"
        self.output_shape = output_shape
        self.input_shape = None

    def initialize_weights(self):
        """Reshape layer has no weights to initialize."""
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Reshape input to specified shape, keeping batch dimension.

        Parameters:
            input_data: Shape (batch, ...) - any shape with batch as first dimension

        Returns:
            Reshaped array of shape (batch, *output_shape)
        """
        super().forward(input_data)
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]

        # Reshape to (batch_size, *output_shape)
        if np.prod(input_data.shape[1:]) != np.prod(self.output_shape):
            raise ValueError("ReshapeLayer output_shape does not match input features")
        output = input_data.reshape(batch_size, *self.output_shape)

        logger.debug(
            f"Reshape layer {self.name}: input shape {input_data.shape}, output shape {output.shape}"
        )
        return output

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        """
        Reshape gradient back to original input shape.

        Parameters:
            output_gradient: Shape (batch, *output_shape)

        Returns:
            Dictionary with 'inputs' reshaped to original input shape
        """
        input_gradient = output_gradient.reshape(self.input_shape)

        logger.debug(
            f"Reshape layer {self.name} backward: output_gradient shape {output_gradient.shape}, "
            f"input_gradient shape {input_gradient.shape}"
        )

        return LayerGradients(input_gradient=input_gradient, parameter_gradients={})

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "output_shape": self.output_shape,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReshapeLayer":
        return cls(output_shape=tuple(data["output_shape"]), name=data.get("name"))


class MaxPoolLayer(Layer):
    """
    Max pooling layer that reduces spatial dimensions by taking maximum values.
    """

    def __init__(self, pool_size: int = 2, stride: int = 2, name: str = None):
        super().__init__()
        _require_positive_int(pool_size, "pool_size")
        _require_positive_int(stride, "stride")
        self.name = name
        self.type = "MaxPool"
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def initialize_weights(self):
        """MaxPool layer has no weights to initialize."""
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Max pooling forward pass using strided views.

        Parameters:
            input_data: Shape (batch, channels, height, width)

        Returns:
            Pooled output of shape (batch, channels, out_h, out_w)
        """
        if input_data.ndim != 4:
            raise ValueError("MaxPoolLayer expects 4D input")
        if input_data.shape[2] < self.pool_size or input_data.shape[3] < self.pool_size:
            raise ValueError("MaxPoolLayer pool_size exceeds input spatial dimensions")

        super().forward(input_data)
        self.last_input = input_data

        batch_size, channels, height, width = input_data.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1

        # Create strided view of input windows
        # Shape: (batch, out_h, out_w, channels, pool_h, pool_w)
        shape = (batch_size, out_h, out_w, channels, self.pool_size, self.pool_size)
        strides = (
            input_data.strides[0],
            input_data.strides[2] * self.stride,
            input_data.strides[3] * self.stride,
            input_data.strides[1],
            input_data.strides[2],
            input_data.strides[3],
        )

        windows = np.lib.stride_tricks.as_strided(
            input_data, shape=shape, strides=strides
        )

        # Max pool: take max over pool dimensions (last 2 axes)
        output = np.max(windows, axis=(4, 5))  # (batch, out_h, out_w, channels)
        output = np.transpose(output, (0, 3, 1, 2))  # (batch, channels, out_h, out_w)

        logger.debug(
            f"MaxPool layer {self.name}: input shape {input_data.shape}, output shape {output.shape}"
        )
        return output

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        """
        Max pooling backward pass using strided views.

        Parameters:
            output_gradient: Shape (batch, channels, out_h, out_w)

        Returns:
            Dictionary with 'inputs' gradient
        """
        batch_size, channels, height, width = self.last_input.shape
        _, _, out_h, out_w = output_gradient.shape

        # Create strided view of input windows
        shape = (batch_size, out_h, out_w, channels, self.pool_size, self.pool_size)
        strides = (
            self.last_input.strides[0],
            self.last_input.strides[2] * self.stride,
            self.last_input.strides[3] * self.stride,
            self.last_input.strides[1],
            self.last_input.strides[2],
            self.last_input.strides[3],
        )

        windows = np.lib.stride_tricks.as_strided(
            self.last_input, shape=shape, strides=strides
        )

        # Reshape windows for max comparison
        windows_reshaped = windows.reshape(batch_size, out_h, out_w, channels, -1)

        # Find max indices
        max_indices = np.argmax(
            windows_reshaped, axis=4
        )  # (batch, out_h, out_w, channels)

        # Create mask where max values are
        max_mask = np.zeros_like(windows_reshaped)
        np.put_along_axis(max_mask, max_indices[..., np.newaxis], 1, axis=4)

        # Reshape mask back to pool window shape
        max_mask = max_mask.reshape(
            batch_size, out_h, out_w, channels, self.pool_size, self.pool_size
        )

        # Transpose output gradient to match window shape
        grad_transposed = np.transpose(
            output_gradient, (0, 2, 3, 1)
        )  # (batch, out_h, out_w, channels)
        grad_expanded = grad_transposed[
            ..., np.newaxis, np.newaxis
        ]  # (batch, out_h, out_w, channels, 1, 1)

        # Apply mask and redistribute gradients
        grad_windows = grad_expanded * max_mask

        # Accumulate gradients back to input
        input_gradient = np.zeros_like(self.last_input)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                input_gradient[:, :, h_start:h_end, w_start:w_end] += grad_windows[
                    :, i, j, :, :, :
                ]

        logger.debug(
            f"MaxPool layer {self.name} backward: output_gradient shape {output_gradient.shape}, "
            f"input_gradient shape {input_gradient.shape}"
        )

        return LayerGradients(input_gradient=input_gradient, parameter_gradients={})

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "pool_size": self.pool_size,
            "stride": self.stride,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MaxPoolLayer":
        return cls(
            pool_size=data.get("pool_size", 2),
            stride=data.get("stride", 2),
            name=data.get("name"),
        )


class BatchNormLayer(Layer):
    """
    Batch Normalization layer that normalizes inputs and applies learned scale/shift.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
    ):
        super().__init__()
        _require_positive_int(num_features, "num_features")
        self.name = name
        self.type = "BatchNormLayer"
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = None  # scale
        self.beta = None  # shift

        # Running statistics for inference
        self.running_mean = None
        self.running_var = None

        # Cached values for backward pass
        self.x_normalized = None
        self.batch_mean = None
        self.batch_var = None
        self.input_shape = None

    @staticmethod
    def _as_feature_matrix(input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim == 2:
            return input_data
        if input_data.ndim == 4:
            return input_data.transpose(0, 2, 3, 1).reshape(-1, input_data.shape[1])
        raise ValueError("BatchNormLayer expects 2D or 4D input")

    def _restore_input_shape(self, feature_matrix: np.ndarray) -> np.ndarray:
        if len(self.input_shape) == 2:
            return feature_matrix
        batch_size, channels, height, width = self.input_shape
        return feature_matrix.reshape(batch_size, height, width, channels).transpose(
            0, 3, 1, 2
        )

    def initialize_weights(self):
        """Initialize gamma=1, beta=0, and running statistics."""
        self.gamma = np.ones(self.num_features)
        self.beta = np.zeros(self.num_features)
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.ones(self.num_features)
        logger.info(f"Batch norm parameters initialized for layer {self.name}")

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter
            for name, parameter in {"gamma": self.gamma, "beta": self.beta}.items()
            if parameter is not None
        }

    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Batch normalization forward pass.

        Parameters:
            input_data: Shape (batch, features) or (batch, channels, height, width)

        Returns:
            Normalized output of same shape as input
        """
        super().forward(input_data)
        self.last_input = input_data
        self.input_shape = input_data.shape
        batch_data = self._as_feature_matrix(input_data)
        if batch_data.shape[1] != self.num_features:
            raise ValueError("BatchNormLayer input feature dimension does not match num_features")

        if training:
            self.batch_mean = np.mean(batch_data, axis=0)
            self.batch_var = np.var(batch_data, axis=0)
            mean = self.batch_mean
            variance = self.batch_var
        else:
            mean = self.running_mean
            variance = self.running_var

        self.x_normalized = (batch_data - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        output = self.gamma * self.x_normalized + self.beta

        if training:
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * self.batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * self.batch_var
            )

        output = self._restore_input_shape(output)

        logger.debug(
            f"BatchNorm layer {self.name}: input shape {input_data.shape}, output shape {output.shape}"
        )
        return output

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        """
        Batch normalization backward pass.

        Parameters:
            output_gradient: Shape same as forward output

        Returns:
            Dictionary with gradients for inputs, gamma, and beta
        """
        batch_grad = self._as_feature_matrix(output_gradient)

        batch_size = batch_grad.shape[0]

        # Gradient w.r.t. gamma and beta
        gamma_gradient = np.sum(batch_grad * self.x_normalized, axis=0)
        beta_gradient = np.sum(batch_grad, axis=0)

        scaled_gradient = batch_grad * self.gamma
        inverse_std = 1 / np.sqrt(self.batch_var + self.epsilon)
        input_gradient = inverse_std / batch_size * (
            batch_size * scaled_gradient
            - np.sum(scaled_gradient, axis=0)
            - self.x_normalized * np.sum(scaled_gradient * self.x_normalized, axis=0)
        )
        input_gradient = self._restore_input_shape(input_gradient)

        logger.debug(
            f"BatchNorm layer {self.name} backward: output_gradient shape {output_gradient.shape}, "
            f"input_gradient shape {input_gradient.shape}"
        )

        return LayerGradients(
            input_gradient=input_gradient,
            parameter_gradients={"gamma": gamma_gradient, "beta": beta_gradient},
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "num_features": self.num_features,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BatchNormLayer":
        return cls(
            num_features=data["num_features"],
            momentum=data.get("momentum", 0.9),
            epsilon=data.get("epsilon", 1e-5),
            name=data.get("name"),
        )


class LayerNormLayer(Layer):
    """Normalize each input independently across its final feature axis."""

    def __init__(self, num_features: int, epsilon: float = 1e-5, name: str = None):
        super().__init__()
        _require_positive_int(num_features, "num_features")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.name = name
        self.type = "LayerNormLayer"
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.x_normalized = None
        self.inverse_std = None

    def initialize_weights(self):
        self.gamma = np.ones(self.num_features)
        self.beta = np.zeros(self.num_features)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter
            for name, parameter in {"gamma": self.gamma, "beta": self.beta}.items()
            if parameter is not None
        }

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim < 2:
            raise ValueError("LayerNormLayer expects at least 2D input")
        if input_data.shape[-1] != self.num_features:
            raise ValueError(
                "LayerNormLayer input feature dimension does not match num_features"
            )

        super().forward(input_data)
        self.last_input = input_data
        mean = np.mean(input_data, axis=-1, keepdims=True)
        variance = np.var(input_data, axis=-1, keepdims=True)
        self.inverse_std = 1 / np.sqrt(variance + self.epsilon)
        self.x_normalized = (input_data - mean) * self.inverse_std
        return self.gamma * self.x_normalized + self.beta

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        if output_gradient.shape != self.last_input.shape:
            raise ValueError("LayerNormLayer gradient has the wrong shape")

        parameter_axes = tuple(range(output_gradient.ndim - 1))
        gamma_gradient = np.sum(output_gradient * self.x_normalized, axis=parameter_axes)
        beta_gradient = np.sum(output_gradient, axis=parameter_axes)
        normalized_gradient = output_gradient * self.gamma
        input_gradient = self.inverse_std * (
            normalized_gradient
            - np.mean(normalized_gradient, axis=-1, keepdims=True)
            - self.x_normalized
            * np.mean(
                normalized_gradient * self.x_normalized, axis=-1, keepdims=True
            )
        )
        return LayerGradients(
            input_gradient=input_gradient,
            parameter_gradients={"gamma": gamma_gradient, "beta": beta_gradient},
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "num_features": self.num_features,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LayerNormLayer":
        return cls(
            num_features=data["num_features"],
            epsilon=data.get("epsilon", 1e-5),
            name=data.get("name"),
        )


class EmbeddingLayer(Layer):
    """Map integer token IDs to vectors from a trainable lookup table.

    The backward pass returns ``None`` for input gradients because token IDs are
    discrete. Gradients for repeated token IDs accumulate in the lookup table.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, name: str = None):
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive")
        self.name = name
        self.type = "EmbeddingLayer"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def initialize_weights(self):
        self.weights = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        logger.info(f"Embedding weights initialized for layer {self.name}")

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim != 2:
            raise ValueError("EmbeddingLayer expects 2D token IDs")
        if not np.issubdtype(input_data.dtype, np.integer):
            raise ValueError("EmbeddingLayer expects integer token IDs")
        if np.any(input_data < 0) or np.any(input_data >= self.vocab_size):
            raise ValueError("EmbeddingLayer token IDs are out of range")

        super().forward(input_data)
        self.last_input = input_data
        return self.weights[input_data]

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        expected_shape = self.last_input.shape + (self.embedding_dim,)
        if output_gradient.shape != expected_shape:
            raise ValueError("EmbeddingLayer gradient has the wrong shape")

        weight_gradient = np.zeros_like(self.weights)
        np.add.at(weight_gradient, self.last_input, output_gradient)
        return LayerGradients(
            input_gradient=None, parameter_gradients={"weights": weight_gradient}
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingLayer":
        return cls(
            vocab_size=data["vocab_size"],
            embedding_dim=data["embedding_dim"],
            name=data.get("name"),
        )


class PositionalEncodingLayer(Layer):
    """Add fixed sinusoidal position information to sequence vectors."""

    def __init__(self, embedding_dim: int, name: str = None):
        super().__init__()
        _require_positive_int(embedding_dim, "embedding_dim")
        self.name = name
        self.type = "PositionalEncodingLayer"
        self.embedding_dim = embedding_dim
        self.encoding = None

    def initialize_weights(self):
        """Positional encodings are fixed and have no trainable arrays."""

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim != 3:
            raise ValueError("PositionalEncodingLayer expects 3D input")
        if not np.issubdtype(input_data.dtype, np.floating):
            raise ValueError("PositionalEncodingLayer expects floating-point input")
        if input_data.shape[-1] != self.embedding_dim:
            raise ValueError(
                "PositionalEncodingLayer input feature dimension does not match embedding_dim"
            )

        super().forward(input_data)
        self.last_input = input_data
        sequence_length = input_data.shape[1]
        positions = np.arange(sequence_length)[:, np.newaxis]
        frequencies = np.exp(
            np.arange(0, self.embedding_dim, 2) * -np.log(10000.0) / self.embedding_dim
        )
        self.encoding = np.zeros((sequence_length, self.embedding_dim))
        self.encoding[:, 0::2] = np.sin(positions * frequencies)
        self.encoding[:, 1::2] = np.cos(
            positions * frequencies[: self.encoding[:, 1::2].shape[1]]
        )
        return input_data + self.encoding[np.newaxis, :, :]

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        if output_gradient.shape != self.last_input.shape:
            raise ValueError("PositionalEncodingLayer gradient has the wrong shape")
        return LayerGradients(input_gradient=output_gradient, parameter_gradients={})

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PositionalEncodingLayer":
        return cls(
            embedding_dim=data["embedding_dim"],
            name=data.get("name"),
        )


class DotProductAttentionLayer(Layer):
    """Single-head self-attention without positional or causal masking."""

    def __init__(self, embedding_dim: int, name: str = None):
        super().__init__()
        _require_positive_int(embedding_dim, "embedding_dim")
        self.name = name
        self.type = "DotProductAttentionLayer"
        self.embedding_dim = embedding_dim
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.output_weights = None
        self.queries = None
        self.keys = None
        self.values = None
        self.scores = None
        self.attention_weights = None
        self.attention_output = None

    def initialize_weights(self):
        scale = np.sqrt(1 / self.embedding_dim)
        shape = (self.embedding_dim, self.embedding_dim)
        self.query_weights = np.random.randn(*shape) * scale
        self.key_weights = np.random.randn(*shape) * scale
        self.value_weights = np.random.randn(*shape) * scale
        self.output_weights = np.random.randn(*shape) * scale

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter
            for name, parameter in {
                "query_weights": self.query_weights,
                "key_weights": self.key_weights,
                "value_weights": self.value_weights,
                "output_weights": self.output_weights,
            }.items()
            if parameter is not None
        }

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exponentials = np.exp(shifted_scores)
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim != 3:
            raise ValueError("DotProductAttentionLayer expects 3D input")
        if not np.issubdtype(input_data.dtype, np.floating):
            raise ValueError("DotProductAttentionLayer expects floating-point input")
        if input_data.shape[-1] != self.embedding_dim:
            raise ValueError(
                "DotProductAttentionLayer input feature dimension does not match embedding_dim"
            )

        super().forward(input_data)
        self.last_input = input_data
        self.queries = input_data @ self.query_weights
        self.keys = input_data @ self.key_weights
        self.values = input_data @ self.value_weights
        self.scores = self.queries @ np.swapaxes(self.keys, -1, -2)
        self.scores /= np.sqrt(self.embedding_dim)
        self.attention_weights = self._softmax(self.scores)
        self.attention_output = self.attention_weights @ self.values
        return self.attention_output @ self.output_weights

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        if output_gradient.shape != self.last_input.shape:
            raise ValueError("DotProductAttentionLayer gradient has the wrong shape")

        output_weights_gradient = np.einsum(
            "bld,ble->de", self.attention_output, output_gradient
        )
        attention_output_gradient = output_gradient @ self.output_weights.T

        attention_weights_gradient = attention_output_gradient @ np.swapaxes(
            self.values, -1, -2
        )
        value_gradient = np.swapaxes(self.attention_weights, -1, -2) @ (
            attention_output_gradient
        )

        score_gradient = self.attention_weights * (
            attention_weights_gradient
            - np.sum(
                attention_weights_gradient * self.attention_weights,
                axis=-1,
                keepdims=True,
            )
        )
        scale = np.sqrt(self.embedding_dim)
        query_gradient = score_gradient @ self.keys / scale
        key_gradient = np.swapaxes(score_gradient, -1, -2) @ self.queries / scale

        query_weights_gradient = np.einsum(
            "bld,ble->de", self.last_input, query_gradient
        )
        key_weights_gradient = np.einsum(
            "bld,ble->de", self.last_input, key_gradient
        )
        value_weights_gradient = np.einsum(
            "bld,ble->de", self.last_input, value_gradient
        )
        input_gradient = (
            query_gradient @ self.query_weights.T
            + key_gradient @ self.key_weights.T
            + value_gradient @ self.value_weights.T
        )

        return LayerGradients(
            input_gradient=input_gradient,
            parameter_gradients={
                "query_weights": query_weights_gradient,
                "key_weights": key_weights_gradient,
                "value_weights": value_weights_gradient,
                "output_weights": output_weights_gradient,
            },
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DotProductAttentionLayer":
        return cls(
            embedding_dim=data["embedding_dim"],
            name=data.get("name"),
        )


class TransformerBlock(Layer):
    """Single-head attention, residual connections, normalization, and feed-forward layers.

    The computation remains explicit:
    ``x -> attention -> + x -> layer norm -> dense -> GeLU -> dense -> + -> layer norm``.
    """

    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int | None = None,
        name: str = None,
    ):
        super().__init__()
        _require_positive_int(embedding_dim, "embedding_dim")
        if feed_forward_dim is None:
            feed_forward_dim = 4 * embedding_dim
        _require_positive_int(feed_forward_dim, "feed_forward_dim")
        self.name = name
        self.type = "TransformerBlock"
        self.embedding_dim = embedding_dim
        self.feed_forward_dim = feed_forward_dim
        self.attention = DotProductAttentionLayer(embedding_dim, name="attention")
        self.layer_norm_1 = LayerNormLayer(embedding_dim, name="layer_norm_1")
        self.feed_forward_1 = DenseLayer(
            embedding_dim, feed_forward_dim, GeLU(), name="feed_forward_1"
        )
        self.feed_forward_2 = DenseLayer(
            feed_forward_dim,
            embedding_dim,
            DifferentiableFunction(lambda x: x, lambda x: np.ones_like(x)),
            name="feed_forward_2",
        )
        self.layer_norm_2 = LayerNormLayer(embedding_dim, name="layer_norm_2")
        self.residual_1 = None
        self.residual_2 = None

    def _named_layers(self) -> Dict[str, Layer]:
        return {
            "attention": self.attention,
            "layer_norm_1": self.layer_norm_1,
            "feed_forward_1": self.feed_forward_1,
            "feed_forward_2": self.feed_forward_2,
            "layer_norm_2": self.layer_norm_2,
        }

    def initialize_weights(self):
        for layer in self._named_layers().values():
            layer.initialize_weights()
            layer.weights_initialized = True

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            f"{layer_name}.{parameter_name}": parameter
            for layer_name, layer in self._named_layers().items()
            for parameter_name, parameter in layer.parameters().items()
        }

    def set_parameter(self, name: str, value: np.ndarray) -> None:
        layer_name, parameter_name = name.split(".", maxsplit=1)
        if layer_name not in self._named_layers():
            raise ValueError(f"Unknown TransformerBlock parameter: {name}")
        self._named_layers()[layer_name].set_parameter(parameter_name, value)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim != 3:
            raise ValueError("TransformerBlock expects 3D input")
        if not np.issubdtype(input_data.dtype, np.floating):
            raise ValueError("TransformerBlock expects floating-point input")
        if input_data.shape[-1] != self.embedding_dim:
            raise ValueError(
                "TransformerBlock input feature dimension does not match embedding_dim"
            )

        super().forward(input_data)
        self.last_input = input_data
        attention_output = self.attention.forward(input_data)
        self.residual_1 = input_data + attention_output
        normalized_attention = self.layer_norm_1.forward(self.residual_1)
        feed_forward_hidden = self.feed_forward_1.forward(normalized_attention)
        feed_forward_output = self.feed_forward_2.forward(feed_forward_hidden)
        self.residual_2 = normalized_attention + feed_forward_output
        return self.layer_norm_2.forward(self.residual_2)

    def backward(self, output_gradient: np.ndarray) -> LayerGradients:
        if output_gradient.shape != self.last_input.shape:
            raise ValueError("TransformerBlock gradient has the wrong shape")

        layer_norm_2_gradients = self.layer_norm_2.backward(output_gradient)
        feed_forward_2_gradients = self.feed_forward_2.backward(
            layer_norm_2_gradients.input_gradient
        )
        feed_forward_1_gradients = self.feed_forward_1.backward(
            feed_forward_2_gradients.input_gradient
        )
        layer_norm_1_gradients = self.layer_norm_1.backward(
            layer_norm_2_gradients.input_gradient
            + feed_forward_1_gradients.input_gradient
        )
        attention_gradients = self.attention.backward(layer_norm_1_gradients.input_gradient)

        parameter_gradients = {}
        for layer_name, gradients in {
            "attention": attention_gradients,
            "layer_norm_1": layer_norm_1_gradients,
            "feed_forward_1": feed_forward_1_gradients,
            "feed_forward_2": feed_forward_2_gradients,
            "layer_norm_2": layer_norm_2_gradients,
        }.items():
            parameter_gradients.update(
                {
                    f"{layer_name}.{parameter_name}": gradient
                    for parameter_name, gradient in gradients.parameter_gradients.items()
                }
            )

        return LayerGradients(
            input_gradient=layer_norm_1_gradients.input_gradient
            + attention_gradients.input_gradient,
            parameter_gradients=parameter_gradients,
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "embedding_dim": self.embedding_dim,
            "feed_forward_dim": self.feed_forward_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TransformerBlock":
        return cls(
            embedding_dim=data["embedding_dim"],
            feed_forward_dim=data.get("feed_forward_dim"),
            name=data.get("name"),
        )
