import unittest
import numpy as np
from Model import Model
from Layer import DenseLayer
from DifferentiableFunction import DifferentiableFunction

class DummyLoss(DifferentiableFunction):
    def __init__(self):
        def function(x):
            return x ** 2
        def derivative(x):
            return 2 * x
        super().__init__(function, derivative)

class DummyLayer(DenseLayer):
    def __init__(self):
        self.last_input = None
        self.last_grad = None
    def forward(self, x):
        self.last_input = x
        return x + 1
    def backward(self, grad, lr):
        self.last_grad = grad
        return grad * 0.5

class TestModel(unittest.TestCase):
    def setUp(self):
        self.layers = [DummyLayer(), DummyLayer()]
        self.loss = DummyLoss()
        self.model = Model(self.layers, self.loss)
        self.x = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y_true = np.array([[2.0, 3.0], [4.0, 5.0]])

    def test_forward(self):
        out = self.model.forward(self.x)
        np.testing.assert_array_equal(out, self.x + 2)

    def test_predict(self):
        out = self.model.predict(self.x)
        np.testing.assert_array_equal(out, self.x + 2)

    def test_compute_loss(self):
        y_pred = self.model.forward(self.x)
        loss = self.model.compute_loss(self.y_true, y_pred)
        expected = np.mean((y_pred - self.y_true) ** 2)
        self.assertAlmostEqual(loss, expected)

    def test_backward_calls_layers(self):
        y_pred = self.model.forward(self.x)
        self.model.backward(self.y_true, y_pred, learning_rate=0.1)
        for layer in self.layers:
            self.assertIsNotNone(layer.last_grad)

if __name__ == "__main__":
    unittest.main()