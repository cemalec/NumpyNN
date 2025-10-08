import numpy as np
from DifferentiableFunction import DifferentiableFunction

class DenseLayer:
    def __init__(self, 
                 input_size:int,
                 output_size: int, 
                 activation_function: DifferentiableFunction):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
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

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
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

        # Update weights and biases
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient
    