import numpy as np
from Layer import DenseLayer, Layer, CNNLayer, FlattenLayer, ReshapeLayer
from DifferentiableFunction import ReLU


class DummyActivation:
    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class DummyOptimizer:
    def __init__(self):
        self.learning_rate = 0.1

    def update(self, layer: Layer, dW, db):
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
    new_weights = np.array([[0.99, 0.98], [0.98, 0.96], [0.97, 0.94]])
    new_biases = np.array([-0.01, -0.02])
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

    new_weights = np.array([[0.74, 0.7], [0.62, 0.56]])
    new_biases = np.array([-0.12, -0.14])

    delta = output_gradient * np.ones_like(layer.last_z)
    expected_weights_gradient = layer.last_input.T @ delta
    expected_biases_gradient = np.sum(delta, axis=0)
    layer = optimizer.update(layer, expected_weights_gradient, expected_biases_gradient)
    np.testing.assert_allclose(
        layer.weights, new_weights
    )
    np.testing.assert_allclose(
        layer.biases, new_biases
    )


def test_cnn_layer_forward():
    """Test CNN layer forward pass with simple input."""
    # Create a simple 1-channel 5x5 input
    batch_size = 2
    input_data = np.arange(batch_size * 1 * 5 * 5).reshape(batch_size, 1, 5, 5).astype(float)

    activation = ReLU()
    layer = CNNLayer(
        input_size=(1, 5, 5),
        output_size=(2, 3, 3),
        kernel_size=3,
        num_filters=2,
        padding=0,
        stride=1,
        name="test_cnn"
    )

    output = layer.forward(input_data)

    # Output shape should be (batch_size, num_filters, out_h, out_w)
    # (2, 5, 5) -> (2, 2, 3, 3) with kernel_size=3, stride=1, padding=0
    assert output.shape == (batch_size, 2, 3, 3), f"Expected shape (2, 2, 3, 3), got {output.shape}"
    assert layer.weights_initialized is True


def test_cnn_layer_backward():
    """Test CNN layer backward pass."""
    batch_size = 2
    input_data = np.random.randn(batch_size, 1, 5, 5).astype(float)

    activation = ReLU()
    layer = CNNLayer(
        input_size=(1, 5, 5),
        output_size=(2, 3, 3),
        kernel_size=3,
        num_filters=2,
        padding=0,
        stride=1,
        name="test_cnn"
    )

    # Forward pass
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    grad_dict = layer.backward(output_gradient)

    # Check gradient shapes
    assert grad_dict["inputs"].shape == input_data.shape, \
        f"Input gradient shape {grad_dict['inputs'].shape} != input shape {input_data.shape}"
    assert grad_dict["weights"].shape == layer.weights.shape, \
        f"Weight gradient shape {grad_dict['weights'].shape} != weights shape {layer.weights.shape}"
    assert grad_dict["biases"].shape == layer.biases.shape, \
        f"Bias gradient shape {grad_dict['biases'].shape} != biases shape {layer.biases.shape}"


def test_flatten_layer_forward():
    """Test Flatten layer forward pass."""
    batch_size = 4
    input_data = np.random.randn(batch_size, 3, 8, 8).astype(float)

    layer = FlattenLayer(name="test_flatten")
    output = layer.forward(input_data)

    # Output shape should be (batch_size, 3*8*8)
    expected_shape = (batch_size, 3 * 8 * 8)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def test_flatten_layer_backward():
    """Test Flatten layer backward pass."""
    batch_size = 4
    input_data = np.random.randn(batch_size, 3, 8, 8).astype(float)

    layer = FlattenLayer(name="test_flatten")
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    grad_dict = layer.backward(output_gradient)

    # Input gradient should match original input shape
    assert grad_dict["inputs"].shape == input_data.shape, \
        f"Input gradient shape {grad_dict['inputs'].shape} != input shape {input_data.shape}"
    assert grad_dict["weights"] is None
    assert grad_dict["biases"] is None


def test_reshape_layer_forward():
    """Test Reshape layer forward pass."""
    batch_size = 8
    input_data = np.random.randn(batch_size, 784).astype(float)

    layer = ReshapeLayer(output_shape=(1, 28, 28), name="test_reshape")
    output = layer.forward(input_data)

    # Output shape should be (batch_size, 1, 28, 28)
    expected_shape = (batch_size, 1, 28, 28)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def test_reshape_layer_backward():
    """Test Reshape layer backward pass."""
    batch_size = 8
    input_data = np.random.randn(batch_size, 784).astype(float)

    layer = ReshapeLayer(output_shape=(1, 28, 28), name="test_reshape")
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    grad_dict = layer.backward(output_gradient)

    # Input gradient should match original input shape
    assert grad_dict["inputs"].shape == input_data.shape, \
        f"Input gradient shape {grad_dict['inputs'].shape} != input shape {input_data.shape}"
    assert grad_dict["weights"] is None
    assert grad_dict["biases"] is None


def test_cnn_to_dict_and_from_dict():
    """Test CNN layer serialization."""
    layer = CNNLayer(
        input_size=(1, 28, 28),
        output_size=(32, 26, 26),
        kernel_size=3,
        num_filters=32,
        padding=0,
        stride=1,
        name="conv1"
    )

    layer_dict = layer.to_dict()
    restored_layer = CNNLayer.from_dict(layer_dict)

    assert restored_layer.name == layer.name
    assert restored_layer.type == layer.type
    assert restored_layer.input_size == layer.input_size
    assert restored_layer.output_size == layer.output_size
    assert restored_layer.kernel_size == layer.kernel_size
    assert restored_layer.num_filters == layer.num_filters
    assert restored_layer.padding == layer.padding
    assert restored_layer.stride == layer.stride


def test_flatten_to_dict_and_from_dict():
    """Test Flatten layer serialization."""
    layer = FlattenLayer(name="flatten1")

    layer_dict = layer.to_dict()
    restored_layer = FlattenLayer.from_dict(layer_dict)

    assert restored_layer.name == layer.name
    assert restored_layer.type == layer.type


def test_reshape_to_dict_and_from_dict():
    """Test Reshape layer serialization."""
    layer = ReshapeLayer(output_shape=(1, 28, 28), name="reshape1")

    layer_dict = layer.to_dict()
    restored_layer = ReshapeLayer.from_dict(layer_dict)

    assert restored_layer.name == layer.name
    assert restored_layer.type == layer.type
    assert restored_layer.output_shape == layer.output_shape
