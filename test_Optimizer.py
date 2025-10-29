import numpy as np
import pytest
import numpy as np
import pytest
from Optimizer import SGD, RMSProp, Adam

class DummyLayer:
    def __init__(self, name, weights, biases):
        self.name = name
        self.weights = weights
        self.biases = biases

def test_sgd_step_updates_weights_and_biases():
    w0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    b0 = np.array([0.5, -0.5], dtype=float)
    grads = {
        'weights': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=float),
        'biases': np.array([0.01, -0.02], dtype=float)
    }
    layer = DummyLayer('sgdLayer', w0.copy(), b0.copy())
    opt = SGD(learning_rate=0.1)

    opt.step(layer, grads)

    expected_w = w0 - 0.1 * grads['weights']
    expected_b = b0 - 0.1 * grads['biases']

    np.testing.assert_allclose(layer.weights, expected_w, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(layer.biases, expected_b, rtol=1e-7, atol=1e-12)

def test_rmsprop_initializes_state_and_performs_two_updates():
    w0 = np.array([[0.5, -0.2]], dtype=float)
    b0 = np.array([0.0], dtype=float)
    grads = {
        'weights': np.array([[0.2, -0.1]], dtype=float),
        'biases': np.array([0.05], dtype=float)
    }
    layer = DummyLayer('rmsLayer', w0.copy(), b0.copy())
    lr = 0.01
    beta = 0.9
    eps = 1e-8
    opt = RMSProp(learning_rate=lr, beta=beta, epsilon=eps)

    # First update
    opt.step(layer, grads)
    # expected s after first update (initial s is zero)
    s1_w = (1 - beta) * (grads['weights'] ** 2)
    s1_b = (1 - beta) * (grads['biases'] ** 2)
    expected_w1 = w0 - lr * grads['weights'] / (np.sqrt(s1_w) + eps)
    expected_b1 = b0 - lr * grads['biases'] / (np.sqrt(s1_b) + eps)

    assert layer.name in opt.s
    np.testing.assert_allclose(opt.s[layer.name]['weights'], s1_w, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(opt.s[layer.name]['biases'], s1_b, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(layer.weights, expected_w1, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(layer.biases, expected_b1, rtol=1e-8, atol=1e-12)

    # Second update (accumulate s and apply update again)
    prev_w = layer.weights.copy()
    prev_b = layer.biases.copy()
    opt.step(layer, grads)

    s2_w = beta * s1_w + (1 - beta) * (grads['weights'] ** 2)
    s2_b = beta * s1_b + (1 - beta) * (grads['biases'] ** 2)
    expected_w2 = prev_w - lr * grads['weights'] / (np.sqrt(s2_w) + eps)
    expected_b2 = prev_b - lr * grads['biases'] / (np.sqrt(s2_b) + eps)

    np.testing.assert_allclose(opt.s[layer.name]['weights'], s2_w, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(opt.s[layer.name]['biases'], s2_b, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(layer.weights, expected_w2, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(layer.biases, expected_b2, rtol=1e-8, atol=1e-12)

# --- Adam tests added below ---

def simulate_adam_updates(weights, biases, grads, lr, beta1, beta2, eps, steps):
    m = {'weights': np.zeros_like(weights), 'biases': np.zeros_like(biases)}
    v = {'weights': np.zeros_like(weights), 'biases': np.zeros_like(biases)}
    t = 0
    w = weights.copy()
    b = biases.copy()
    for _ in range(steps):
        t += 1
        for key, param in (('weights', w), ('biases', b)):
            g = grads[key]
            m[key] = beta1 * m[key] + (1 - beta1) * g
            v[key] = beta2 * v[key] + (1 - beta2) * (g ** 2)
            m_hat = m[key] / (1 - beta1 ** t)
            v_hat = v[key] / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            if key == 'weights':
                w = param - update
            else:
                b = param - update
    return w, b, m, v, t

def test_adam_step_single_update():
    w0 = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=float)
    b0 = np.array([0.1, -0.1], dtype=float)
    grads = {
        'weights': np.array([[0.1, -0.2], [0.3, -0.4]], dtype=float),
        'biases': np.array([0.01, -0.02], dtype=float)
    }
    layer = DummyLayer('adamL1', w0.copy(), b0.copy())
    opt = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    opt.step(layer, grads)

    expected_w, expected_b, expected_m, expected_v, expected_t = simulate_adam_updates(
        w0, b0, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=1
    )

    assert opt.t == expected_t
    assert layer.name in opt.m and layer.name in opt.v
    np.testing.assert_allclose(layer.weights, expected_w, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(layer.biases, expected_b, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(opt.m[layer.name]['weights'], expected_m['weights'])
    np.testing.assert_allclose(opt.v[layer.name]['weights'], expected_v['weights'])

def test_adam_step_two_updates():
    w0 = np.array([[0.2, -0.3]], dtype=float)
    b0 = np.array([0.05], dtype=float)
    grads = {
        'weights': np.array([[0.05, -0.07]], dtype=float),
        'biases': np.array([0.005], dtype=float)
    }
    layer = DummyLayer('adamLayerX', w0.copy(), b0.copy())
    opt = Adam(learning_rate=0.005, beta1=0.85, beta2=0.995, epsilon=1e-7)

    # Apply two steps using the optimizer under test
    opt.step(layer, grads)
    opt.step(layer, grads)

    # Compute expected after two steps
    expected_w, expected_b, expected_m, expected_v, expected_t = simulate_adam_updates(
        w0, b0, grads, lr=0.005, beta1=0.85, beta2=0.995, eps=1e-7, steps=2
    )

    assert opt.t == expected_t
    np.testing.assert_allclose(layer.weights, expected_w, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(layer.biases, expected_b, rtol=1e-6, atol=1e-9)
    assert layer.name in opt.m and layer.name in opt.v
    np.testing.assert_allclose(opt.m[layer.name]['biases'], expected_m['biases'])
    np.testing.assert_allclose(opt.v[layer.name]['biases'], expected_v['biases'])
