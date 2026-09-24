import numpy as np
import pytest
from Layer import (
    DenseLayer,
    Layer,
    CNNLayer,
    FlattenLayer,
    MaxPoolLayer,
    DotProductAttentionLayer,
    PositionalEncodingLayer,
    ReshapeLayer,
    BatchNormLayer,
    EmbeddingLayer,
    LayerNormLayer,
    TransformerBlock,
)
from DifferentiableFunction import GeLU, ReLU


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
    np.testing.assert_allclose(layer.weights, new_weights)
    np.testing.assert_allclose(layer.biases, new_biases)


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
    np.testing.assert_allclose(layer.weights, new_weights)
    np.testing.assert_allclose(layer.biases, new_biases)


def test_dense_layer_applies_its_map_to_each_sequence_position():
    layer = DenseLayer(2, 2, DummyActivation())
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.biases = np.array([0.5, -0.5])
    layer.weights_initialized = True
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]])

    output = layer.forward(inputs)
    gradients = layer.backward(np.ones_like(output))

    np.testing.assert_allclose(
        output, np.array([[[1.5, 1.5], [3.5, 3.5]]])
    )
    np.testing.assert_allclose(
        gradients.parameter_gradients["weights"], np.array([[1.0, 1.0], [1.0, 1.0]])
    )
    np.testing.assert_allclose(gradients.input_gradient.shape, inputs.shape)


def test_cnn_layer_forward():
    """Test CNN layer forward pass with simple input."""
    # Create a simple 1-channel 5x5 input
    batch_size = 2
    input_data = (
        np.arange(batch_size * 1 * 5 * 5).reshape(batch_size, 1, 5, 5).astype(float)
    )

    layer = CNNLayer(
        input_size=(1, 5, 5),
        output_size=(2, 3, 3),
        kernel_size=3,
        num_filters=2,
        padding=0,
        stride=1,
        name="test_cnn",
    )

    output = layer.forward(input_data)

    # Output shape should be (batch_size, num_filters, out_h, out_w)
    # (2, 5, 5) -> (2, 2, 3, 3) with kernel_size=3, stride=1, padding=0
    assert output.shape == (
        batch_size,
        2,
        3,
        3,
    ), f"Expected shape (2, 2, 3, 3), got {output.shape}"
    assert layer.weights_initialized is True


def test_cnn_layer_backward():
    """Test CNN layer backward pass."""
    batch_size = 2
    input_data = np.random.randn(batch_size, 1, 5, 5).astype(float)

    layer = CNNLayer(
        input_size=(1, 5, 5),
        output_size=(2, 3, 3),
        kernel_size=3,
        num_filters=2,
        padding=0,
        stride=1,
        name="test_cnn",
    )

    # Forward pass
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    gradients = layer.backward(output_gradient)

    # Check gradient shapes
    assert (
        gradients.input_gradient.shape == input_data.shape
    ), f"Input gradient shape {gradients.input_gradient.shape} != input shape {input_data.shape}"
    assert (
        gradients.parameter_gradients["weights"].shape == layer.weights.shape
    ), f"Weight gradient shape {gradients.parameter_gradients['weights'].shape} != weights shape {layer.weights.shape}"
    assert (
        gradients.parameter_gradients["biases"].shape == layer.biases.shape
    ), f"Bias gradient shape {gradients.parameter_gradients['biases'].shape} != biases shape {layer.biases.shape}"


def test_cnn_padding_and_activation():
    layer = CNNLayer(
        input_size=(1, 3, 4),
        output_size=(1, 5, 6),
        kernel_size=1,
        num_filters=1,
        padding=1,
        activation_function=ReLU(),
        name="padded_conv",
    )
    layer.weights = np.array([[[[1.0]]]])
    layer.biases = np.array([-1.0])
    layer.weights_initialized = True

    output = layer.forward(np.ones((2, 1, 3, 4)))
    gradients = layer.backward(np.ones_like(output))

    assert output.shape == (2, 1, 5, 6)
    assert np.all(output >= 0)
    assert gradients.input_gradient.shape == (2, 1, 3, 4)


def test_flatten_layer_forward():
    """Test Flatten layer forward pass."""
    batch_size = 4
    input_data = np.random.randn(batch_size, 3, 8, 8).astype(float)

    layer = FlattenLayer(name="test_flatten")
    output = layer.forward(input_data)

    # Output shape should be (batch_size, 3*8*8)
    expected_shape = (batch_size, 3 * 8 * 8)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


def test_flatten_layer_backward():
    """Test Flatten layer backward pass."""
    batch_size = 4
    input_data = np.random.randn(batch_size, 3, 8, 8).astype(float)

    layer = FlattenLayer(name="test_flatten")
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    gradients = layer.backward(output_gradient)

    # Input gradient should match original input shape
    assert (
        gradients.input_gradient.shape == input_data.shape
    ), f"Input gradient shape {gradients.input_gradient.shape} != input shape {input_data.shape}"
    assert gradients.parameter_gradients == {}


def test_reshape_layer_forward():
    """Test Reshape layer forward pass."""
    batch_size = 8
    input_data = np.random.randn(batch_size, 784).astype(float)

    layer = ReshapeLayer(output_shape=(1, 28, 28), name="test_reshape")
    output = layer.forward(input_data)

    # Output shape should be (batch_size, 1, 28, 28)
    expected_shape = (batch_size, 1, 28, 28)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


def test_reshape_layer_backward():
    """Test Reshape layer backward pass."""
    batch_size = 8
    input_data = np.random.randn(batch_size, 784).astype(float)

    layer = ReshapeLayer(output_shape=(1, 28, 28), name="test_reshape")
    output = layer.forward(input_data)

    # Create output gradient
    output_gradient = np.random.randn(*output.shape).astype(float)

    # Backward pass
    gradients = layer.backward(output_gradient)

    # Input gradient should match original input shape
    assert (
        gradients.input_gradient.shape == input_data.shape
    ), f"Input gradient shape {gradients.input_gradient.shape} != input shape {input_data.shape}"
    assert gradients.parameter_gradients == {}


def test_cnn_to_dict_and_from_dict():
    """Test CNN layer serialization."""
    layer = CNNLayer(
        input_size=(1, 28, 28),
        output_size=(32, 26, 26),
        kernel_size=3,
        num_filters=32,
        padding=0,
        stride=1,
        name="conv1",
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


def test_dense_layer_serializes_gelu_activation():
    layer = DenseLayer(2, 3, GeLU(), name="gelu_dense")

    restored_layer = DenseLayer.from_dict(layer.to_dict())

    assert isinstance(restored_layer.activation_function, GeLU)


def test_dense_layer_rejects_unknown_activation_type():
    with pytest.raises(ValueError, match="Unsupported activation type"):
        DenseLayer.from_dict(
            {
                "type": "Dense",
                "name": "invalid",
                "input_size": 2,
                "output_size": 1,
                "activation_function": "Unknown",
            }
        )


@pytest.mark.parametrize(
    ("input_size", "output_size"), [(0, 1), (1, 0), (-1, 1)]
)
def test_dense_layer_validates_layer_sizes(input_size, output_size):
    with pytest.raises(ValueError, match="positive integer"):
        DenseLayer(input_size, output_size, ReLU())


def test_dense_layer_validates_input_feature_dimension():
    layer = DenseLayer(2, 1, ReLU())

    with pytest.raises(ValueError, match="feature dimension"):
        layer.forward(np.ones((3, 4)))


def test_cnn_and_pool_validate_configuration_and_input_shapes():
    with pytest.raises(ValueError, match="kernel_size"):
        CNNLayer((1, 3, 3), (1, 1, 1), kernel_size=0, num_filters=1)
    with pytest.raises(ValueError, match="pool_size"):
        MaxPoolLayer(pool_size=0)

    layer = CNNLayer((1, 3, 3), (1, 1, 1), kernel_size=3, num_filters=1)
    with pytest.raises(ValueError, match="channel dimension"):
        layer.forward(np.ones((1, 2, 3, 3)))


def test_batchnorm_validates_feature_dimension():
    layer = BatchNormLayer(num_features=2)

    with pytest.raises(ValueError, match="feature dimension"):
        layer.forward(np.ones((3, 4)))


def test_batchnorm_layer_forward():
    """Test BatchNorm layer forward pass."""
    batch_size = 4
    num_features = 3
    input_data = np.random.randn(batch_size, num_features).astype(float)

    layer = BatchNormLayer(num_features=num_features, name="test_bn")
    output = layer.forward(input_data)

    # Output shape should match input shape
    assert (
        output.shape == input_data.shape
    ), f"Expected shape {input_data.shape}, got {output.shape}"
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
    assert (
        output.shape == input_data.shape
    ), f"Expected shape {input_data.shape}, got {output.shape}"


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
    gradients = layer.backward(output_gradient)

    # Check gradient shapes
    assert (
        gradients.input_gradient.shape == input_data.shape
    ), f"Input gradient shape {gradients.input_gradient.shape} != input shape {input_data.shape}"
    assert gradients.parameter_gradients["gamma"].shape == (
        num_features,
    ), f"Gamma gradient shape {gradients.parameter_gradients['gamma'].shape} != expected (num_features,)"
    assert gradients.parameter_gradients["beta"].shape == (
        num_features,
    ), f"Beta gradient shape {gradients.parameter_gradients['beta'].shape} != expected (num_features,)"


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
    gradients = layer.backward(output_gradient)

    # Check gradient shapes
    assert (
        gradients.input_gradient.shape == input_data.shape
    ), f"Input gradient shape {gradients.input_gradient.shape} != input shape {input_data.shape}"
    assert gradients.parameter_gradients["gamma"].shape == (
        channels,
    ), f"Gamma gradient shape {gradients.parameter_gradients['gamma'].shape} != expected ({channels},)"
    assert gradients.parameter_gradients["beta"].shape == (
        channels,
    ), f"Beta gradient shape {gradients.parameter_gradients['beta'].shape} != expected ({channels},)"


def test_batchnorm_4d_input_gradient_matches_finite_difference():
    np.random.seed(1)
    inputs = np.random.randn(2, 3, 2, 2)
    output_gradient = np.random.randn(*inputs.shape)
    layer = BatchNormLayer(num_features=3, momentum=1.0, name="test_bn_gradient")

    layer.forward(inputs)
    analytic_gradient = layer.backward(output_gradient).input_gradient
    numerical_gradient = np.zeros_like(inputs)
    epsilon = 1e-6
    for index in np.ndindex(inputs.shape):
        positive = inputs.copy()
        negative = inputs.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        numerical_gradient[index] = (
            np.sum(layer.forward(positive) * output_gradient)
            - np.sum(layer.forward(negative) * output_gradient)
        ) / (2 * epsilon)

    np.testing.assert_allclose(analytic_gradient, numerical_gradient, atol=1e-6)


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

    layer = BatchNormLayer(
        num_features=num_features, momentum=0.0, name="test_bn_scale"
    )

    # Set specific gamma and beta
    layer.gamma = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    layer.beta = np.array([1.0, 0.5, -0.5, 2.0, -1.0])

    output = layer.forward(input_data)

    # The output should be scaled and shifted version of normalized input
    # output = gamma * normalized_input + beta
    expected_output = layer.gamma * layer.x_normalized + layer.beta
    np.testing.assert_allclose(output, expected_output, rtol=1e-6)


def test_batchnorm_running_statistics():
    """Test that BatchNorm updates running statistics."""
    batch_size = 16
    num_features = 4
    momentum = 0.9

    layer = BatchNormLayer(
        num_features=num_features, momentum=momentum, name="test_bn_running"
    )

    # First batch
    input_data_1 = np.random.randn(batch_size, num_features) + 1.0

    batch_mean_1 = np.mean(input_data_1, axis=0)
    batch_var_1 = np.var(input_data_1, axis=0)

    expected_running_mean_1 = momentum * 0 + (1 - momentum) * batch_mean_1
    expected_running_var_1 = momentum * 1 + (1 - momentum) * batch_var_1
    _ = layer.forward(input_data_1)
    np.testing.assert_allclose(layer.running_mean, expected_running_mean_1, rtol=1e-6)
    np.testing.assert_allclose(layer.running_var, expected_running_var_1, rtol=1e-6)

    # Second batch
    input_data_2 = np.random.randn(batch_size, num_features) - 1.0

    batch_mean_2 = np.mean(input_data_2, axis=0)
    batch_var_2 = np.var(input_data_2, axis=0)

    expected_running_mean_2 = (
        momentum * expected_running_mean_1 + (1 - momentum) * batch_mean_2
    )
    expected_running_var_2 = (
        momentum * expected_running_var_1 + (1 - momentum) * batch_var_2
    )
    _ = layer.forward(input_data_2)
    np.testing.assert_allclose(layer.running_mean, expected_running_mean_2, rtol=1e-6)
    np.testing.assert_allclose(layer.running_var, expected_running_var_2, rtol=1e-6)


def test_batchnorm_to_dict_and_from_dict():
    """Test BatchNorm layer serialization."""
    layer = BatchNormLayer(num_features=32, momentum=0.9, epsilon=1e-5, name="bn1")

    layer_dict = layer.to_dict()
    restored_layer = BatchNormLayer.from_dict(layer_dict)

    assert restored_layer.name == layer.name
    assert restored_layer.type == layer.type
    assert restored_layer.num_features == layer.num_features
    assert restored_layer.momentum == layer.momentum
    assert restored_layer.epsilon == layer.epsilon


def test_layernorm_normalizes_each_sequence_position():
    layer = LayerNormLayer(num_features=3)
    inputs = np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]])

    output = layer.forward(inputs)

    np.testing.assert_allclose(np.mean(output, axis=-1), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        output,
        (inputs - np.mean(inputs, axis=-1, keepdims=True))
        / np.sqrt(np.var(inputs, axis=-1, keepdims=True) + layer.epsilon),
    )


def test_layernorm_backward_matches_finite_differences():
    layer = LayerNormLayer(num_features=2)
    layer.gamma = np.array([1.2, -0.7])
    layer.beta = np.array([0.3, -0.2])
    layer.weights_initialized = True
    inputs = np.array([[[0.2, -0.4], [0.7, 0.3]]])
    output_gradient = np.array([[[0.5, -0.2], [-0.3, 0.4]]])
    layer.forward(inputs)
    gradients = layer.backward(output_gradient)
    epsilon = 1e-6

    def loss() -> float:
        return np.sum(layer.forward(inputs) * output_gradient)

    for parameter_name, index in {"gamma": 1, "beta": 0}.items():
        parameter = getattr(layer, parameter_name)
        original_value = parameter[index]
        parameter[index] = original_value + epsilon
        positive_loss = loss()
        parameter[index] = original_value - epsilon
        negative_loss = loss()
        parameter[index] = original_value
        numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)
        np.testing.assert_allclose(
            gradients.parameter_gradients[parameter_name][index],
            numerical_gradient,
            rtol=1e-5,
            atol=1e-6,
        )

    input_index = (0, 1, 0)
    original_input = inputs[input_index]
    inputs[input_index] = original_input + epsilon
    positive_loss = loss()
    inputs[input_index] = original_input - epsilon
    negative_loss = loss()
    inputs[input_index] = original_input
    numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)
    np.testing.assert_allclose(
        gradients.input_gradient[input_index],
        numerical_gradient,
        rtol=1e-5,
        atol=1e-6,
    )


def test_layernorm_validates_input_contract():
    layer = LayerNormLayer(num_features=2)

    with pytest.raises(ValueError, match="at least 2D"):
        layer.forward(np.ones(2))
    with pytest.raises(ValueError, match="feature dimension"):
        layer.forward(np.ones((1, 2, 3)))


def test_embedding_layer_looks_up_token_vectors():
    layer = EmbeddingLayer(vocab_size=4, embedding_dim=2, name="tokens")
    layer.weights = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    layer.weights_initialized = True

    output = layer.forward(np.array([[2, 0], [1, 2]]))

    np.testing.assert_array_equal(
        output,
        np.array([[[4.0, 5.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]]),
    )


def test_embedding_layer_accumulates_repeated_token_gradients():
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    layer.weights = np.zeros((3, 2))
    layer.weights_initialized = True
    layer.forward(np.array([[1, 1, 2]]))

    gradients = layer.backward(np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]))

    assert gradients.input_gradient is None
    np.testing.assert_array_equal(
        gradients.parameter_gradients["weights"],
        np.array([[0.0, 0.0], [4.0, 6.0], [5.0, 6.0]]),
    )


def test_embedding_layer_validates_token_ids():
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)

    with pytest.raises(ValueError, match="integer"):
        layer.forward(np.array([[1.0]]))
    with pytest.raises(ValueError, match="out of range"):
        layer.forward(np.array([[3]]))


def test_embedding_weight_gradient_matches_finite_difference():
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    layer.weights = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    layer.weights_initialized = True
    token_ids = np.array([[0, 1, 0]])
    output_gradient = np.array([[[0.7, -0.1], [0.2, 0.5], [-0.3, 0.4]]])
    layer.forward(token_ids)
    analytic_gradient = layer.backward(output_gradient).parameter_gradients["weights"]
    epsilon = 1e-6
    index = (0, 1)
    original_value = layer.weights[index]

    layer.weights[index] = original_value + epsilon
    positive_loss = np.sum(layer.forward(token_ids) * output_gradient)
    layer.weights[index] = original_value - epsilon
    negative_loss = np.sum(layer.forward(token_ids) * output_gradient)
    layer.weights[index] = original_value

    numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)
    np.testing.assert_allclose(analytic_gradient[index], numerical_gradient)


def test_attention_forward_shape_and_attention_rows():
    layer = DotProductAttentionLayer(embedding_dim=3)
    inputs = np.array([[[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]])

    output = layer.forward(inputs)

    assert output.shape == inputs.shape
    np.testing.assert_allclose(np.sum(layer.attention_weights, axis=-1), 1.0)
    assert set(layer.parameters()) == {
        "query_weights",
        "key_weights",
        "value_weights",
        "output_weights",
    }


def test_positional_encoding_distinguishes_identical_tokens_by_position():
    layer = PositionalEncodingLayer(embedding_dim=4)
    inputs = np.zeros((1, 3, 4))

    output = layer.forward(inputs)

    np.testing.assert_allclose(output[0, 0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert not np.allclose(output[0, 0], output[0, 1])
    assert layer.parameters() == {}


def test_positional_encoding_backward_is_identity():
    layer = PositionalEncodingLayer(embedding_dim=2)
    layer.forward(np.ones((1, 2, 2)))
    output_gradient = np.array([[[0.2, -0.1], [0.4, 0.3]]])

    gradients = layer.backward(output_gradient)

    np.testing.assert_array_equal(gradients.input_gradient, output_gradient)
    assert gradients.parameter_gradients == {}


@pytest.mark.parametrize(
    "inputs, message",
    [
        (np.ones((1, 2)), "3D"),
        (np.ones((1, 2, 2), dtype=int), "floating-point"),
        (np.ones((1, 2, 3)), "feature dimension"),
    ],
)
def test_positional_encoding_validates_input_contract(inputs, message):
    with pytest.raises(ValueError, match=message):
        PositionalEncodingLayer(embedding_dim=2).forward(inputs)


def test_attention_is_permutation_equivariant_without_positions():
    layer = DotProductAttentionLayer(embedding_dim=2)
    layer.query_weights = np.array([[1.0, 0.5], [0.0, 1.0]])
    layer.key_weights = np.array([[0.5, 0.0], [1.0, 1.0]])
    layer.value_weights = np.array([[1.0, -0.5], [0.5, 1.0]])
    layer.output_weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.weights_initialized = True
    inputs = np.array([[[1.0, 2.0], [3.0, 1.0], [0.0, -1.0]]])
    permutation = np.array([2, 0, 1])

    output = layer.forward(inputs)
    permuted_output = layer.forward(inputs[:, permutation])

    np.testing.assert_allclose(permuted_output, output[:, permutation])


@pytest.mark.parametrize(
    "inputs, message",
    [
        (np.ones((2, 3)), "3D"),
        (np.ones((1, 2, 3), dtype=int), "floating-point"),
        (np.ones((1, 2, 4)), "feature dimension"),
    ],
)
def test_attention_validates_input_contract(inputs, message):
    with pytest.raises(ValueError, match=message):
        DotProductAttentionLayer(embedding_dim=3).forward(inputs)


def test_attention_backward_matches_finite_differences():
    layer = DotProductAttentionLayer(embedding_dim=2)
    layer.query_weights = np.array([[0.2, -0.3], [0.4, 0.1]])
    layer.key_weights = np.array([[-0.2, 0.5], [0.3, 0.2]])
    layer.value_weights = np.array([[0.1, 0.4], [-0.5, 0.2]])
    layer.output_weights = np.array([[0.3, -0.1], [0.2, 0.6]])
    layer.weights_initialized = True
    inputs = np.array([[[0.2, -0.4], [0.7, 0.3]]])
    output_gradient = np.array([[[0.5, -0.2], [-0.3, 0.4]]])

    layer.forward(inputs)
    gradients = layer.backward(output_gradient)
    epsilon = 1e-6

    def loss() -> float:
        return np.sum(layer.forward(inputs) * output_gradient)

    for parameter_name, index in {
        "query_weights": (0, 1),
        "key_weights": (1, 0),
        "value_weights": (0, 0),
        "output_weights": (1, 1),
    }.items():
        parameter = getattr(layer, parameter_name)
        original_value = parameter[index]
        parameter[index] = original_value + epsilon
        positive_loss = loss()
        parameter[index] = original_value - epsilon
        negative_loss = loss()
        parameter[index] = original_value
        numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)
        np.testing.assert_allclose(
            gradients.parameter_gradients[parameter_name][index],
            numerical_gradient,
            rtol=1e-5,
            atol=1e-6,
        )

    input_index = (0, 1, 0)
    original_input = inputs[input_index]
    inputs[input_index] = original_input + epsilon
    positive_loss = loss()
    inputs[input_index] = original_input - epsilon
    negative_loss = loss()
    inputs[input_index] = original_input
    numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)
    np.testing.assert_allclose(
        gradients.input_gradient[input_index],
        numerical_gradient,
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_backward_validates_gradient_shape():
    layer = DotProductAttentionLayer(embedding_dim=2)
    layer.forward(np.ones((1, 2, 2)))

    with pytest.raises(ValueError, match="wrong shape"):
        layer.backward(np.ones((1, 2, 1)))


def test_transformer_block_returns_input_and_parameter_gradients():
    block = TransformerBlock(embedding_dim=2, feed_forward_dim=3)
    inputs = np.array([[[0.2, -0.1], [0.4, 0.3]]])

    output = block.forward(inputs)
    gradients = block.backward(np.ones_like(output))

    assert output.shape == inputs.shape
    assert gradients.input_gradient.shape == inputs.shape
    assert set(gradients.parameter_gradients) == set(block.parameters())
    assert all(
        gradient.shape == block.parameters()[name].shape
        for name, gradient in gradients.parameter_gradients.items()
    )
