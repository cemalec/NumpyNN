import numpy as np
from Layer import DenseLayer

class DummyActivation:
    def function(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)

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
    learning_rate = 0.5

    # Save old weights and biases for manual calculation
    old_weights = layer.weights.copy()
    old_biases = layer.biases.copy()

    input_grad = layer.backward(output_gradient, learning_rate)

    # Calculate expected gradients
    activation_derivative = np.ones_like(layer.last_z)
    delta = output_gradient * activation_derivative
    expected_weights_gradient = layer.last_input.T @ delta
    expected_biases_gradient = np.sum(delta, axis=0)
    expected_input_gradient = delta @ old_weights.T

    # Check weights and biases update
    np.testing.assert_allclose(
        layer.weights,
        old_weights - learning_rate * expected_weights_gradient
    )
    np.testing.assert_allclose(
        layer.biases,
        old_biases - learning_rate * expected_biases_gradient
    )

    # Check input gradient
    np.testing.assert_allclose(input_grad, expected_input_gradient)

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
    learning_rate = 0.1

    old_weights = layer.weights.copy()
    old_biases = layer.biases.copy()

    input_grad = layer.backward(output_gradient, learning_rate)

    delta = output_gradient * np.ones_like(layer.last_z)
    expected_weights_gradient = layer.last_input.T @ delta
    expected_biases_gradient = np.sum(delta, axis=0)
    expected_input_gradient = delta @ old_weights.T

    np.testing.assert_allclose(
        layer.weights,
        old_weights - learning_rate * expected_weights_gradient
    )
    np.testing.assert_allclose(
        layer.biases,
        old_biases - learning_rate * expected_biases_gradient
    )
    np.testing.assert_allclose(input_grad, expected_input_gradient)