import unittest
import numpy as np
from Model import Model
from Layer import BatchNormLayer, DenseLayer, EmbeddingLayer, LayerGradients
from Optimizer import Optimizer, SGD
from DifferentiableFunction import CrossEntropyLoss, DifferentiableFunction, SoftMax


dummy_loss = DifferentiableFunction(
    lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size,
)

dummy_activation = DifferentiableFunction(lambda x: x + 1, lambda x: np.ones_like(x))


class DummyLayer(DenseLayer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return inputs + 1  # simple operation for testing

    def backward(self, grad_outputs: np.ndarray) -> LayerGradients:
        grad_inputs = grad_outputs  # pass gradient unchanged
        return LayerGradients(input_gradient=grad_inputs, parameter_gradients={})


dummy_layer = DummyLayer(2, 2, activation_function=dummy_activation)

dummy_optimizer = Optimizer()


class TestModel(unittest.TestCase):
    def setUp(self):
        self.layers = [dummy_layer, dummy_layer]
        self.loss = dummy_loss
        self.optimizer = dummy_optimizer
        self.model = Model(self.layers, self.loss, self.optimizer)
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
        self.model.backward(self.y_true, y_pred)
        for layer in self.layers:
            self.assertTrue(hasattr(layer, "inputs"))

    def test_softmax_cross_entropy_uses_logit_gradient(self):
        layer = DenseLayer(2, 2, SoftMax(), name="output")
        layer.weights = np.array([[0.3, -0.2], [0.1, 0.4]])
        layer.biases = np.array([0.05, -0.05])
        layer.weights_initialized = True
        model = Model([layer], CrossEntropyLoss(), SGD(learning_rate=0.1))
        inputs = np.array([[0.2, -0.1], [0.4, 0.3]])
        targets = np.array([[0.0, 1.0], [1.0, 0.0]])

        predictions = model.forward(inputs)
        expected_gradient = inputs.T @ ((predictions - targets) / len(inputs))
        expected_weights = layer.weights - 0.1 * expected_gradient

        model.backward(targets, predictions)

        np.testing.assert_allclose(layer.weights, expected_weights)


def test_predict_uses_batchnorm_running_statistics_without_updating_them():
    batch_norm = BatchNormLayer(num_features=2, momentum=0.5, name="batch_norm")
    model = Model([batch_norm], CrossEntropyLoss(), SGD(learning_rate=0.1))
    model.forward(np.array([[1.0, 3.0], [5.0, 7.0]]))
    running_mean = batch_norm.running_mean.copy()
    running_var = batch_norm.running_var.copy()
    inputs = np.array([[9.0, 11.0]])

    predictions = model.predict(inputs)

    expected = (inputs - running_mean) / np.sqrt(running_var + batch_norm.epsilon)
    np.testing.assert_allclose(predictions, expected)
    np.testing.assert_array_equal(batch_norm.running_mean, running_mean)
    np.testing.assert_array_equal(batch_norm.running_var, running_var)


def test_model_save_and_load_uses_layer_parameters(tmp_path):
    batch_norm = BatchNormLayer(num_features=2, name="batch_norm")
    batch_norm.gamma = np.array([2.0, 3.0])
    batch_norm.beta = np.array([-1.0, 0.5])
    batch_norm.weights_initialized = True
    model = Model([batch_norm], CrossEntropyLoss(), SGD(learning_rate=0.1))
    path = tmp_path / "batch_norm_model.npz"

    model.save(str(path))
    restored = Model.load(str(path))

    assert set(batch_norm.parameters()) == {"gamma", "beta"}
    np.testing.assert_array_equal(restored.layers[0].gamma, batch_norm.gamma)
    np.testing.assert_array_equal(restored.layers[0].beta, batch_norm.beta)


def test_model_save_and_load_preserves_embedding_weights(tmp_path):
    embedding = EmbeddingLayer(vocab_size=3, embedding_dim=2, name="tokens")
    embedding.weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    embedding.weights_initialized = True
    model = Model([embedding], CrossEntropyLoss(), SGD(learning_rate=0.1))
    path = tmp_path / "embedding_model.npz"

    model.save(str(path))
    restored = Model.load(str(path))

    assert isinstance(restored.layers[0], EmbeddingLayer)
    np.testing.assert_array_equal(restored.layers[0].weights, embedding.weights)


if __name__ == "__main__":
    unittest.main()
