import numpy as np
from Layer import DenseLayer,Layer


class DummyActivation:
    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

class DummyOptimizer:
    def __init__(self):
        self.learning_rate = 0.1
    def update(self, layer:Layer, dW, db):
        learning_rate = 0.1
        layer.weights -= learning_rate * dW
        layer.biases -= learning_rate * db
        return layer

def test_backward_updates_weights_and_biases():
    input_size = 3
    output_size = 2
    activation = DummyActivation()
    layer = DenseLayer(input_size, output_size, activation)

    # Set known weights, biases, input, and last_z
    layer.weights = np.ones((input_size, output_size))
    layer.biases = np.zeros(output_size)
    layer.last_input = np.array([[1.0, 2.0, 3.0]])
    layer.last_z = np.array([[0.5, -0.5]])

    output_gradient = np.array([[0.1, 0.2]])
    new_weights = np.array([[0.99,0.98],[0.98,0.96],[0.97,0.94]])
    new_biases = np.array([-0.01,-0.02])
    optimizer = DummyOptimizer()
    # Calculate expected gradients
    activation_derivative = np.ones_like(layer.last_z)
    delta = output_gradient * activation_derivative
    expected_weights_gradient = layer.last_input.T @ delta
    expected_biases_gradient = np.sum(delta, axis=0)
    layer = optimizer.update(layer, expected_weights_gradient, expected_biases_gradient)

    # Check weights and biases update
    np.testing.assert_allclose(
        layer.weights, new_weights
    )
    np.testing.assert_allclose(
        layer.biases, new_biases
    )


def test_backward_with_multiple_samples():
    input_size = 2
    output_size = 2
    activation = DummyActivation()
    layer = DenseLayer(input_size, output_size, activation)

    layer.weights = np.ones((input_size, output_size))
    layer.biases = np.zeros(output_size)
    layer.last_input = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.last_z = np.array([[0.1, 0.2], [0.3, 0.4]])

    output_gradient = np.array([[0.5, 0.6], [0.7, 0.8]])
    optimizer = DummyOptimizer()

    new_weights = np.array([[0.74,0.7],[0.62,0.56]])
    new_biases = np.array([-0.12,-0.14])


    delta = output_gradient * np.ones_like(layer.last_z)
    expected_weights_gradient = layer.last_input.T @ delta
    expected_biases_gradient = np.sum(delta, axis=0)
    layer = optimizer.update(layer, expected_weights_gradient, expected_biases_gradient)
    np.testing.assert_allclose(
        layer.weights,new_weights
    )
    np.testing.assert_allclose(
        layer.biases, new_biases
    )
