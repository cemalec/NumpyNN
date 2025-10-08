import numpy as np
import pytest
from DifferentiableFunction import DifferentiableFunction, SoftMax, ReLU, Sigmoid

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
    def f(x): return x**2
    def df(x): return 2*x
    func = DifferentiableFunction(f, df)
    x = np.array([1.0, 2.0, 3.0])
    assert np.all(func.function(x) == x**2)
    assert np.all(func.derivative(x) == 2*x)