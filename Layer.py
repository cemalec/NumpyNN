import numpy as np
from DifferentiableFunction import DifferentiableFunction
from Optimizer import Optimizer
from typing import Dict

class DenseLayer:
    def __init__(self, 
                 input_size:int,
                 output_size: int, 
                 activation_function: DifferentiableFunction,
                 name: str = None,
                 regularization: str = None,
                 lambda_reg: float = 0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.name = name
        self.lambda_reg = lambda_reg
        self.regularization = regularization
        self.weights = np.random.randn(self.input_size, self.output_size) * (np.sqrt(2./self.input_size))
        self.biases = np.zeros(self.output_size)
        self.last_input = None
        self.last_z = None

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

    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer) -> np.ndarray:
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

        if self.regularization == 'l2':
            weight_gradient += self.lambda_reg * self.weights
            bias_gradient += self.lambda_reg * self.biases
        elif self.regularization == 'l1':
            weight_gradient += self.lambda_reg * np.sign(self.weights)
            bias_gradient += self.lambda_reg * np.sign(self.biases)
        else:
            pass  # No regularization
        input_gradient = np.dot(delta, self.weights.T)  # (batch_size, input_size)
        
        self.update_weights(optimizer=optimizer,
                            grads={'weights': weight_gradient, 'biases': bias_gradient})
        
        return input_gradient
    
    def update_weights(self,
                       optimizer: Optimizer,
                       grads: Dict[str,np.ndarray]):
        """
        Updates the weights and biases of the layer using the provided gradients.
        Parameters:
            weight_gradient (np.ndarray): Gradient of the loss with respect to the weights. Shape: (input_size, output_size)
            bias_gradient (np.ndarray): Gradient of the loss with respect to the biases. Shape: (output_size,)
            learning_rate (float): Learning rate for weight updates.
        """
        self.weights,self.biases = optimizer.step(self, grads)