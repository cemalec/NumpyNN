import numpy as np
from DifferentiableFunction import (
    CrossEntropyLoss,
    DifferentiableFunction,
    GeLU,
    SoftMax,
    ReLU,
    Sigmoid,
)


def test_softmax_function():
    sm = SoftMax()
    x = np.array([1.0, 2.0, 3.0])
    result = sm.function(x)
    expected = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_softmax_derivative_shape():
    sm = SoftMax()
    x = np.array([1.0, 2.0, 3.0])
    deriv = sm.derivative(x)
    assert deriv.shape == x.shape


def test_relu_function():
    relu = ReLU()
    x = np.array([-1.0, 0.0, 2.0])
    result = relu.function(x)
    expected = np.maximum(0, x)
    np.testing.assert_array_equal(result, expected)


def test_relu_derivative():
    relu = ReLU()
    x = np.array([-1.0, 0.0, 2.0])
    deriv = relu.derivative(x)
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(deriv, expected)


def test_gelu_matches_exact_definition_and_derivative():
    gelu = GeLU()
    values = np.array([-1.5, -0.25, 0.0, 0.75])
    epsilon = 1e-6

    numerical_derivative = (
        gelu.function(values + epsilon) - gelu.function(values - epsilon)
    ) / (2 * epsilon)

    np.testing.assert_allclose(
        gelu.derivative(values), numerical_derivative, rtol=1e-6, atol=1e-6
    )


def test_gelu_differs_from_relu_near_zero():
    values = np.array([-0.5, 0.5])

    assert not np.allclose(GeLU().function(values), ReLU().function(values))
    assert not np.allclose(GeLU().derivative(values), ReLU().derivative(values))


def test_sigmoid_function():
    sigmoid = Sigmoid()
    x = np.array([-1.0, 0.0, 1.0])
    result = sigmoid.function(x)
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_sigmoid_derivative():
    sigmoid = Sigmoid()
    x = np.array([-1.0, 0.0, 1.0])
    deriv = sigmoid.derivative(x)
    s = 1 / (1 + np.exp(-x))
    expected = s * (1 - s)
    np.testing.assert_allclose(deriv, expected, rtol=1e-6)


def test_differentiable_function_interface():
    def f(x):
        return x**2

    def df(x):
        return 2 * x

    func = DifferentiableFunction(f, df)
    x = np.array([1.0, 2.0, 3.0])
    assert np.all(func.function(x) == x**2)
    assert np.all(func.derivative(x) == 2 * x)


def test_cross_entropy_handles_sequence_predictions():
    targets = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    predictions = np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]],
        ]
    )
    loss = CrossEntropyLoss()

    result = loss.function(targets, predictions)
    gradient = loss.derivative(targets, predictions)

    np.testing.assert_allclose(result, -np.mean(np.log([0.7, 0.8, 0.5, 0.6])))
    np.testing.assert_allclose(gradient, (predictions - targets) / 4)
