from typing import Tuple
import numpy as np
from DifferentiableFunction import DifferentiableFunction
from typing import Dict
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)

class Layer:
    def __init__(self):
        self.name = None
        self.type = 'Layer'
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
    def backward(self, output_gradient: np.ndarray) -> Dict[str,np.ndarray]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> 'Layer':
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

    def __init__(self, 
                 input_size:int,
                 output_size: int, 
                 activation_function: DifferentiableFunction,
                 name: str = None):
        super().__init__()
        self.name = name
        self.type = 'Dense'
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def initialize_weights(self):
        """
        initialize weights with He initialization.
        """
        self.weights = np.random.randn(self.input_size, self.output_size) * (np.sqrt(2./self.input_size))
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
        self.last_z = np.dot(input_data, self.weights) + self.biases  # (batch_size, output_size)
        logger.debug(f"Forward pass in layer {self.name}: input shape {input_data.shape}, z shape {self.last_z.shape}")
        return self.activation_function.function(self.last_z)

    def backward(self, dL_da: np.ndarray) -> Dict[str,np.ndarray]:
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
        da_dz = self.activation_function.derivative(self.last_z)  # (batch_size, output_size)
        # The gradient of the loss with respect to the scores
        dL_dz = dL_da * da_dz  # (batch_size, output_size)
        # The gradient of the scores with respect to weights, biases, and inputs
        dz_dW = self.last_input  # (batch_size, input_size)
        dz_db = 1  # Bias gradient is summed over batch
        dz_di = self.weights  # (input_size, output_size)

        # The gradent of the loss with respect to *this* layer's weights, biases, and inputs
        weight_gradient = np.dot(dz_dW.T, dL_dz) / batches  # (input_size, output_size), used for weight update
        bias_gradient = np.sum(dL_dz*dz_db, axis=0) / batches # (output_size,), used for bias update
        input_gradient = np.dot(dL_dz, dz_di.T)  # (batch_size, input_size), passed to previous layer

        logger.debug(f"Backward pass in layer {self.name}: output_gradient shape {dL_da.shape}, input_gradient shape {input_gradient.shape}") 
        grad_dict = {'inputs': input_gradient,
                     'weights': weight_gradient,
                     'biases': bias_gradient}
        
        return grad_dict

        # Update weights and biases will be handled by the optimizer in Model.py

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation_function': self.activation_function.__class__.__name__,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DenseLayer':
        activation_function = getattr(__import__('DifferentiableFunction'), data['activation_function'])()
        layer = cls(input_size=data['input_size'],
                    output_size=data['output_size'],
                    activation_function=activation_function,
                    name=data['name'])
        return layer
    
class CNNLayer(Layer):
    def __init__(self,
                 input_size: Tuple[int, int],
                 output_size: Tuple[int, int],
                 filter_size: int,
                 num_filters: int,
                 padding: int = 0,
                 stride: int = 1,
                 name: str = None):
        super().__init__()
        self.name = name
        self.type = 'CNN'
        self.input_size = input_size
        self.output_size = output_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride

    def initialize_weights(self):
        self.weights = np.random.randn(self.num_filters,
                                       self.filter_size,
                                       self.filter_size) * 0.01
        self.biases = np.zeros(self.num_filters)
        logger.info(f"Weights and biases initialized for CNN layer {self.name}")

    def pad_input(self, input_data: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            return np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        return input_data

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        input_data = self.pad_input(input_data)
        # Implement the forward pass
        batch_size, _, height, width = input_data.shape
        output_height = (height - self.filter_size) // self.stride + 1
        output_width = (width - self.filter_size) // self.stride + 1
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                input_slice = input_data[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.tensordot(input_slice, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.biases
        return output

    def backward(self, output_gradient: np.ndarray) -> Dict[str,np.ndarray]:
        # Implement the backward pass
        batch_size, _, output_height, output_width = output_gradient.shape
        input_gradient = np.zeros((batch_size, self.num_filters, output_height, output_width))
        weight_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.biases)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                input_slice = output_gradient[:, :, i, j]
                weight_gradient += np.tensordot(input_slice, self.weights, axes=([1], [0]))
                bias_gradient += np.sum(input_slice, axis=0)

        return {
            'inputs': input_gradient,
            'weights': weight_gradient,
            'biases': bias_gradient
        }

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'filter_size': self.filter_size,
            'num_filters': self.num_filters,
            'padding': self.padding,
            'stride': self.stride
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CNNLayer':
        layer = cls(
            filter_size=data['filter_size'],
            num_filters=data['num_filters'],
            padding=data['padding'],
            stride=data['stride']
        )
        return layer