from typing import Tuple, List
import numpy as np
from DifferentiableFunction import DifferentiableFunction
from typing import Dict
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


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
    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
        pass

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
        super().forward(input_data)
        self.last_input = input_data
        self.last_z = (
            np.dot(input_data, self.weights) + self.biases
        )  # (batch_size, output_size)
        logger.debug(
            f"Forward pass in layer {self.name}: input shape {input_data.shape}, z shape {self.last_z.shape}"
        )
        return self.activation_function.function(self.last_z)

    def backward(self, dL_da: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Performs the backward pass through the layer, updating weights and biases.
        Parameters:
            dL_da (np.ndarray): Gradient of the loss with respect to the layer's output. Shape: (batch_size, output_size)
            learning_rate (float): Learning rate for weight updates.
        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input. Shape: (batch_size, input_size)
        """
        batches = self.last_input.shape[0]
        # The gradient of the activation function with respect to the scores (last_z)
        da_dz = self.activation_function.derivative(
            self.last_z
        )  # (batch_size, output_size)
        # The gradient of the loss with respect to the scores
        dL_dz = dL_da * da_dz  # (batch_size, output_size)
        # The gradient of the scores with respect to weights, biases, and inputs
        dz_dW = self.last_input  # (batch_size, input_size)
        dz_db = 1  # Bias gradient is summed over batch
        dz_di = self.weights  # (input_size, output_size)

        # The gradent of the loss with respect to *this* layer's weights, biases, and inputs
        weight_gradient = (
            np.dot(dz_dW.T, dL_dz) / batches
        )  # (input_size, output_size), used for weight update
        bias_gradient = (
            np.sum(dL_dz * dz_db, axis=0) / batches
        )  # (output_size,), used for bias update
        input_gradient = np.dot(
            dL_dz, dz_di.T
        )  # (batch_size, input_size), passed to previous layer

        logger.debug(
            f"Backward pass in layer {self.name}: output_gradient shape {dL_da.shape}, input_gradient shape {input_gradient.shape}"
        )
        grad_dict = {
            "inputs": input_gradient,
            "weights": weight_gradient,
            "biases": bias_gradient,
        }

        return grad_dict

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
        activation_function = getattr(
            __import__("DifferentiableFunction"), data["activation_function"]
        )()
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
        name: str = None,
    ):
        super().__init__()
        self.name = name
        self.type = "CNN"
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride

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
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
                mode="constant",
            )
        return input_data

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        super().forward(input_data)
        input_data = self.pad_input(input_data)

        batch_size, in_channels, height, width = input_data.shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1

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
        output = output + self.biases
        output = np.transpose(output, (0, 3, 1, 2))

        self.last_input = input_data
        return output

    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized backward pass for CNN layer.

        Parameters:
            output_gradient: Shape (batch, num_filters, out_h, out_w)

        Returns:
            Dictionary with 'inputs', 'weights', and 'biases' gradients
        """
        batch_size, num_filters, output_height, output_width = output_gradient.shape
        _, in_channels, padded_height, padded_width = self.last_input.shape

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
        weight_gradient = np.einsum("bhwf,bhwcij->fcij", dL_dout, windows) / batch_size

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

        return {
            "inputs": input_gradient,
            "weights": weight_gradient,
            "biases": bias_gradient,
        }

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
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CNNLayer":
        layer = cls(
            input_size=data["input_size"],
            output_size=data["output_size"],
            kernel_size=data["kernel_size"],
            num_filters=data["num_filters"],
            padding=data.get("padding", 0),
            stride=data.get("stride", 1),
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

    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
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

        return {
            "inputs": input_gradient,
            "weights": None,
            "biases": None,
        }

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
        output = input_data.reshape(batch_size, *self.output_shape)

        logger.debug(
            f"Reshape layer {self.name}: input shape {input_data.shape}, output shape {output.shape}"
        )
        return output

    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
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

        return {
            "inputs": input_gradient,
            "weights": None,
            "biases": None,
        }

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

    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
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

        return {
            "inputs": input_gradient,
            "weights": None,
            "biases": None,
        }

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

    def initialize_weights(self):
        """Initialize gamma=1, beta=0, and running statistics."""
        self.gamma = np.ones(self.num_features)
        self.beta = np.zeros(self.num_features)
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.ones(self.num_features)
        logger.info(f"Batch norm parameters initialized for layer {self.name}")

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Batch normalization forward pass.

        Parameters:
            input_data: Shape (batch, features) or (batch, channels, height, width)

        Returns:
            Normalized output of same shape as input
        """
        super().forward(input_data)
        self.last_input = input_data

        # Reshape to (batch, features) for computation
        original_shape = input_data.shape
        if len(input_data.shape) == 4:  # (batch, channels, height, width)
            batch_data = input_data.reshape(
                input_data.shape[0], input_data.shape[1], -1
            )
            batch_data = batch_data.transpose(0, 2, 1).reshape(-1, input_data.shape[1])
        else:
            batch_data = input_data

        # Compute batch statistics
        self.batch_mean = np.mean(batch_data, axis=0)
        self.batch_var = np.var(batch_data, axis=0)

        # Normalize
        self.x_normalized = (batch_data - self.batch_mean) / np.sqrt(
            self.batch_var + self.epsilon
        )

        # Scale and shift
        output = self.gamma * self.x_normalized + self.beta

        # Update running statistics
        self.running_mean = (
            self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        )
        self.running_var = (
            self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        )

        # Reshape back to original shape
        if len(original_shape) == 4:
            output = output.reshape(
                -1, original_shape[1], original_shape[2], original_shape[3]
            )
        else:
            output = output.reshape(original_shape)

        logger.debug(
            f"BatchNorm layer {self.name}: input shape {input_data.shape}, output shape {output.shape}"
        )
        return output

    def backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Batch normalization backward pass.

        Parameters:
            output_gradient: Shape same as forward output

        Returns:
            Dictionary with gradients for inputs, gamma, and beta
        """
        original_shape = output_gradient.shape

        # Reshape to (batch, features) for computation
        if len(output_gradient.shape) == 4:
            batch_grad = output_gradient.reshape(
                output_gradient.shape[0], output_gradient.shape[1], -1
            )
            batch_grad = batch_grad.transpose(0, 2, 1).reshape(
                -1, output_gradient.shape[1]
            )
        else:
            batch_grad = output_gradient

        batch_size = batch_grad.shape[0]

        # Gradient w.r.t. gamma and beta
        gamma_gradient = np.sum(batch_grad * self.x_normalized, axis=0) / batch_size
        beta_gradient = np.sum(batch_grad, axis=0) / batch_size

        # Gradient w.r.t. normalized input
        x_norm_grad = batch_grad * self.gamma

        # Gradient w.r.t. variance and mean
        var_grad = (
            np.sum(
                x_norm_grad
                * (self.last_input.reshape(-1, self.num_features) - self.batch_mean)
                * -0.5
                * (self.batch_var + self.epsilon) ** -1.5,
                axis=0,
            )
            / batch_size
        )
        mean_grad = (
            np.sum(x_norm_grad * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0)
            / batch_size
        )
        mean_grad += (
            var_grad
            * np.sum(
                -2 * (self.last_input.reshape(-1, self.num_features) - self.batch_mean),
                axis=0,
            )
            / batch_size
        )

        # Gradient w.r.t. input
        input_gradient = (
            x_norm_grad / np.sqrt(self.batch_var + self.epsilon)
            + var_grad
            * 2
            * (self.last_input.reshape(-1, self.num_features) - self.batch_mean)
            / batch_size
            + mean_grad / batch_size
        )

        # Reshape back
        input_gradient = input_gradient.reshape(original_shape)

        logger.debug(
            f"BatchNorm layer {self.name} backward: output_gradient shape {output_gradient.shape}, "
            f"input_gradient shape {input_gradient.shape}"
        )

        return {
            "inputs": input_gradient,
            "gamma": gamma_gradient,
            "beta": beta_gradient,
        }

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
