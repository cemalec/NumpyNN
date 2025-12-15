import numpy as np
from Layer import DenseLayer, Layer, CNNLayer, FlattenLayer, ReshapeLayer, BatchNormLayer
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


def test_batchnorm_layer_forward():
    """Test BatchNorm layer forward pass."""
    batch_size = 4
    num_features = 3
    input_data = np.random.randn(batch_size, num_features).astype(float)
    
    layer = BatchNormLayer(num_features=num_features, name="test_bn")
    output = layer.forward(input_data)
    
    # Output shape should match input shape
    assert output.shape == input_data.shape, f"Expected shape {input_data.shape}, got {output.shape}"
    assert layer.gamma is not None
    assert layer.beta is not None
    assert layer.running_mean is not None
    assert layer.running_var is not None


def test_batchnorm_layer_forward_4d():
    """Test BatchNorm layer forward pass with 4D input (batch, channels, height, width)."""
    batch_size = 2
    channels = 16
    height = 8
    width = 8
    input_data = np.random.randn(batch_size, channels, height, width).astype(float)
    
    layer = BatchNormLayer(num_features=channels, name="test_bn_4d")
    output = layer.forward(input_data)
    
    # Output shape should match input shape
    assert output.shape == input_data.shape, f"Expected shape {input_data.shape}, got {output.shape}"


def test_batchnorm_layer_backward():
    """Test BatchNorm layer backward pass."""
    batch_size = 4
    num_features = 3
    input_data = np.random.randn(batch_size, num_features).astype(float)
    
    layer = BatchNormLayer(num_features=num_features, name="test_bn")
    output = layer.forward(input_data)
    
    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)
    
    # Backward pass
    grad_dict = layer.backward(output_gradient)
    
    # Check gradient shapes
    assert grad_dict["inputs"].shape == input_data.shape, \
        f"Input gradient shape {grad_dict['inputs'].shape} != input shape {input_data.shape}"
    assert grad_dict["gamma"].shape == (num_features,), \
        f"Gamma gradient shape {grad_dict['gamma'].shape} != expected (num_features,)"
    assert grad_dict["beta"].shape == (num_features,), \
        f"Beta gradient shape {grad_dict['beta'].shape} != expected (num_features,)"


def test_batchnorm_layer_backward_4d():
    """Test BatchNorm layer backward pass with 4D input."""
    batch_size = 2
    channels = 16
    height = 8
    width = 8
    input_data = np.random.randn(batch_size, channels, height, width).astype(float)
    
    layer = BatchNormLayer(num_features=channels, name="test_bn_4d")
    output = layer.forward(input_data)
    
    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)
    
    # Backward pass
    grad_dict = layer.backward(output_gradient)
    
    # Check gradient shapes
    assert grad_dict["inputs"].shape == input_data.shape, \
        f"Input gradient shape {grad_dict['inputs'].shape} != input shape {input_data.shape}"
    assert grad_dict["gamma"].shape == (channels,), \
        f"Gamma gradient shape {grad_dict['gamma'].shape} != expected ({channels},)"
    assert grad_dict["beta"].shape == (channels,), \
        f"Beta gradient shape {grad_dict['beta'].shape} != expected ({channels},)"


def test_batchnorm_normalizes_output():
    """Test that BatchNorm actually normalizes the output."""
    batch_size = 32
    num_features = 10
    # Create input with non-zero mean and non-unit variance
    input_data = np.random.randn(batch_size, num_features) * 5 + 3
    
    layer = BatchNormLayer(num_features=num_features, momentum=0.0, name="test_bn_norm")
    output = layer.forward(input_data)
    
    # Check that output is normalized (approximately zero mean, unit variance)
    output_mean = np.mean(output, axis=0)
    output_var = np.var(output, axis=0)
    
    np.testing.assert_allclose(output_mean, 0, atol=1e-5)
    np.testing.assert_allclose(output_var, 1, atol=1e-5)


def test_batchnorm_scale_and_shift():
    """Test that BatchNorm applies scale and shift correctly."""
    batch_size = 16
    num_features = 5
    input_data = np.random.randn(batch_size, num_features)
    
    layer = BatchNormLayer(num_features=num_features, momentum=0.0, name="test_bn_scale")
    
    # Set specific gamma and beta
    layer.gamma = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    layer.beta = np.array([1.0, 0.5, -0.5, 2.0, -1.0])
    
    output = layer.forward(input_data)
    
    # The output should be scaled and shifted version of normalized input
    # output = gamma * normalized_input + beta
    expected_output = (layer.gamma * layer.x_normalized + layer.beta)
    np.testing.assert_allclose(output, expected_output, rtol=1e-6)


def test_batchnorm_running_statistics():
    """Test that BatchNorm updates running statistics."""
    batch_size = 16
    num_features = 4
    momentum = 0.9
    
    layer = BatchNormLayer(num_features=num_features, momentum=momentum, name="test_bn_running")
    
    # First batch
    input_data_1 = np.random.randn(batch_size, num_features) + 1.0
    output_1 = layer.forward(input_data_1)
    
    batch_mean_1 = np.mean(input_data_1, axis=0)
    batch_var_1 = np.var(input_data_1, axis=0)
    
    expected_running_mean_1 = momentum * 0 + (1 - momentum) * batch_mean_1
    expected_running_var_1 = momentum * 1 + (1 - momentum) * batch_var_1
    
    np.testing.assert_allclose(layer.running_mean, expected_running_mean_1, rtol=1e-6)
    np.testing.assert_allclose(layer.running_var, expected_running_var_1, rtol=1e-6)
    
    # Second batch
    input_data_2 = np.random.randn(batch_size, num_features) - 1.0
    output_2 = layer.forward(input_data_2)
    
    batch_mean_2 = np.mean(input_data_2, axis=0)
    batch_var_2 = np.var(input_data_2, axis=0)
    
    expected_running_mean_2 = momentum * expected_running_mean_1 + (1 - momentum) * batch_mean_2
    expected_running_var_2 = momentum * expected_running_var_1 + (1 - momentum) * batch_var_2
    
    np.testing.assert_allclose(layer.running_mean, expected_running_mean_2, rtol=1e-6)
    np.testing.assert_allclose(layer.running_var, expected_running_var_2, rtol=1e-6)


def test_batchnorm_to_dict_and_from_dict():
    """Test BatchNorm layer serialization."""
    layer = BatchNormLayer(
        num_features=32,
        momentum=0.9,
        epsilon=1e-5,
        name="bn1"
    )
    
    layer_dict = layer.to_dict()
    restored_layer = BatchNormLayer.from_dict(layer_dict)
    
    assert restored_layer.name == layer.name
    assert restored_layer.type == layer.type
    assert restored_layer.num_features == layer.num_features
    assert restored_layer.momentum == layer.momentum
    assert restored_layer.epsilon == layer.epsilon
