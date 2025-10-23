import numpy as np
from DifferentiableFunction import DifferentiableFunction
from Optimizer import Optimizer
from typing import Dict
from abc import abstractmethod

class Layer:
    def __init__(self):
        self.name = None
        self.type = 'Layer'
        self.weights = None
        self.biases = None
        self.last_input = None
        self.last_z = None

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass

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

class DenseLayer:
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
        self.initialize_weights()

    def initialize_weights(self):
        """
        initialize weights with He initialization.
        """
        self.weights = np.random.randn(self.input_size, self.output_size) * (np.sqrt(2./self.input_size))
        self.biases = np.zeros(self.output_size)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer, supporting batched input.
        Parameters:
            input_data (np.ndarray): Input data to the layer. Shape: (batch_size, input_size)

        Returns:
            np.ndarray: Output after applying weights, biases, and activation function. Shape: (batch_size, output_size)
        """
        self.last_input = input_data
        self.last_z = np.dot(input_data, self.weights) + self.biases  # (batch_size, output_size)
        return self.activation_function.function(self.last_z)

    def backward(self, output_gradient: np.ndarray) -> Dict[str,np.ndarray]:
        """
        Performs the backward pass through the layer, updating weights and biases.
        Parameters:
            output_gradient (np.ndarray): Gradient of the loss with respect to the layer's output. Shape: (batch_size, output_size)
            learning_rate (float): Learning rate for weight updates.
        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input. Shape: (batch_size, input_size)
            """
        activation_gradient = self.activation_function.derivative(self.last_z)  # (batch_size, output_size)
        delta = output_gradient * activation_gradient  # (batch_size, output_size)

        weight_gradient = np.dot(self.last_input.T, delta) / self.last_input.shape[0]  # (input_size, output_size)
        bias_gradient = np.sum(delta, axis=0) / self.last_input.shape[0] # (output_size,)

        input_gradient = np.dot(delta, self.weights.T)  # (batch_size, input_size)
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