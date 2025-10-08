from typing import Callable
import numpy as np

class DifferentiableFunction:
    function: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]

    def __init__(self, function: Callable[[np.ndarray], np.ndarray], derivative: Callable[[np.ndarray], np.ndarray]):
        self.function = function
        self.derivative = derivative

class SoftMax(DifferentiableFunction):
    def __init__(self):
        def softmax(x: np.ndarray) -> np.ndarray:
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        def softmax_derivative(x: np.ndarray) -> np.ndarray:
            s = softmax(x)
            return s * (1 - s)

        super().__init__(softmax, softmax_derivative)

class ReLU(DifferentiableFunction):
    def __init__(self):
        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

        def relu_derivative(x: np.ndarray) -> np.ndarray:
            return (x > 0).astype(x.dtype)

        super().__init__(relu, relu_derivative)

class Sigmoid(DifferentiableFunction):
    def __init__(self):
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)

class CrossEntropyLoss(DifferentiableFunction):
    def __init__(self):
        def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = y_true.shape[0]
            p = y_pred
            log_likelihood = -np.log(p[range(m), y_true.argmax(axis=1)])
            loss = np.sum(log_likelihood) / m
            return loss

        def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            m = y_true.shape[0]
            grad = y_pred - y_true
            return grad / m

        super().__init__(cross_entropy, cross_entropy_derivative)