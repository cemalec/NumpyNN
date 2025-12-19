import numpy as np
from training_loop import training_loop
import training_loop as tl


class SimpleDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def get_batches(self, batch_size: int, shuffle: bool = True):
        n = len(self.X)
        indices = np.arange(n)
        if shuffle:
            # keep deterministic in tests by not shuffling unless explicitly requested
            pass
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


class LinearDummyModel:
    """
    Simple linear model y = w * x + b trained with plain gradient descent.
    Implements the API expected by training_loop: forward, compute_loss, backward.
    """

    def __init__(self, lr=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = lr
        self.last_input = None
        self.last_pred = None
        self.backward_calls = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X expected shape (batch, 1) or (batch,)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.last_input = X
        pred = (X * self.w).sum(axis=1) + self.b
        self.last_pred = pred
        return pred

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(y_pred.shape)
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true).reshape(y_pred.shape)
        m = y_true.shape[0]
        # gradient of MSE: dL/dpred = 2*(pred - y)/m
        grad_out = 2.0 * (y_pred - y_true) / m
        # grad w = sum(x * grad_out)
        grad_w = (self.last_input.reshape(m, -1)[:, 0] * grad_out).sum()
        grad_b = grad_out.sum()
        # update
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        self.backward_calls += 1


def test_training_loop_decreases_parameter_error():
    np.random.seed(0)
    # generate linear data y = 2*x + 1 with small noise
    n_samples = 200
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    y = (2.0 * X + 1.0 + np.random.normal(0, 0.1, size=(n_samples, 1))).reshape(-1)

    dataset = SimpleDataset(X, y)
    model = LinearDummyModel(lr=0.05)
    # initial param error
    init_err = abs(model.w - 2.0) + abs(model.b - 1.0)

    # training_loop expects an accuracy function; provide a harmless stub
    tl.accuracy = lambda yt, yp: 0.0

    training_loop(model=model, dataset=dataset, epochs=10, batch_size=32)

    final_err = abs(model.w - 2.0) + abs(model.b - 1.0)
    assert (
        final_err < init_err
    ), "Model parameters did not move closer to target after training"


def test_training_loop_calls_backward_expected_number_of_times():
    # small dataset to count backward calls
    X = np.linspace(-1, 1, 50).reshape(-1, 1).astype(np.float32)
    y = (2.0 * X + 1.0).reshape(-1)
    dataset = SimpleDataset(X, y)
    model = LinearDummyModel(lr=0.01)

    tl.accuracy = lambda yt, yp: 0.0

    epochs = 3
    batch_size = 10
    training_loop(model=model, dataset=dataset, epochs=epochs, batch_size=batch_size)

    # expected number of batches per epoch
    expected_batches = int(np.ceil(len(X) / batch_size))
    assert model.backward_calls == expected_batches * epochs
