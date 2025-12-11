from abc import abstractmethod
from typing import Dict, Any
import numpy as np


class Optimizer:
    def __init__(self):
        self.name = None
        self.type = "Optimizer"

    @abstractmethod
    def step(self, layer: Any, grads: Dict[str, np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Optimizer":
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.type = "SGD"

    def step(self, layer, grads: np.ndarray) -> np.ndarray:
        params = {"weights": layer.weights, "biases": layer.biases}
        for key in ["weights", "biases"]:
            params[key] -= self.learning_rate * grads[key]

    def to_dict(self) -> dict:
        return {"learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(learning_rate=data["learning_rate"])


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__()
        self.type = "RMSProp"
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = dict()

    def initialize_state(self, layer: Any):
        self.s[layer.name] = dict()
        self.s[layer.name]["weights"] = np.zeros_like(layer.weights)
        self.s[layer.name]["biases"] = np.zeros_like(layer.biases)

    def step(self, layer: Any, grads: Dict[str, np.ndarray]) -> np.ndarray:
        if self.s.get(layer.name) is None:
            self.initialize_state(layer)
        params = {"weights": layer.weights, "biases": layer.biases}
        for key in ["weights", "biases"]:
            self.s[layer.name][key] = self.beta * self.s[layer.name][key] + (
                1 - self.beta
            ) * (grads[key] ** 2)
            params[key] -= (
                self.learning_rate
                * grads[key]
                / (np.sqrt(self.s[layer.name][key]) + self.epsilon)
            )

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "learning_rate": self.learning_rate,
                "beta": self.beta,
                "epsilon": self.epsilon,
                "type": self.type,
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            learning_rate=data["learning_rate"],
            beta=data.get("beta", 0.9),
            epsilon=data.get("epsilon", 1e-8),
        )


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        m: Dict[str, Any] = None,
        v: Dict[str, Any] = None,
        t: int = 0,
    ):
        super().__init__()
        self.type = "Adam"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        if m is None:
            self.m = dict()
        else:
            self.m = m
        if v is None:
            self.v = dict()
        else:
            self.v = v
        self.t = t

    def initialize_state(self, layer: Any):
        initial_dict = dict(
            weights=np.zeros_like(layer.weights), biases=np.zeros_like(layer.biases)
        )
        self.m[layer.name] = initial_dict.copy()
        self.v[layer.name] = initial_dict.copy()

    def step(self, layer: Any, grads: Dict[str, np.ndarray]) -> np.ndarray:
        if self.m.get(layer.name) is None:
            self.initialize_state(layer)
        params = {"weights": layer.weights, "biases": layer.biases}
        self.t += 1
        for key in ["weights", "biases"]:
            self.m[layer.name][key] = (
                self.beta1 * self.m[layer.name][key] + (1 - self.beta1) * grads[key]
            )
            self.v[layer.name][key] = self.beta2 * self.v[layer.name][key] + (
                1 - self.beta2
            ) * (grads[key] ** 2)
            m_hat = self.m[layer.name][key] / (1 - self.beta1**self.t)
            v_hat = self.v[layer.name][key] / (1 - self.beta2**self.t)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "m": self.m,
            "v": self.v,
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(
            learning_rate=data.get("learning_rate", 0.001),
            beta1=data.get("beta1", 0.9),
            beta2=data.get("beta2", 0.999),
            epsilon=data.get("epsilon", 1e-8),
        )
        return obj

    def save_state(self, filepath: str):
        """Save optimizer state to npz file."""
        np.savez(
            filepath,
            t=self.t,
            m_keys=list(self.m.keys()),
            v_keys=list(self.v.keys()),
            **{f'm_{k}_weights': v['weights'] for k, v in self.m.items()},
            **{f'm_{k}_biases': v['biases'] for k, v in self.m.items()},
            **{f'v_{k}_weights': v['weights'] for k, v in self.v.items()},
            **{f'v_{k}_biases': v['biases'] for k, v in self.v.items()},
        )

    @classmethod
    def load_state(cls, filepath: str, learning_rate: float, **kwargs):
        """Load optimizer state from npz file."""
        data = np.load(filepath, allow_pickle=True)
        m_keys = data['m_keys']
        v_keys = data['v_keys']

        m = {k: {'weights': data[f'm_{k}_weights'], 'biases': data[f'm_{k}_biases']}
             for k in m_keys}
        v = {k: {'weights': data[f'v_{k}_weights'], 'biases': data[f'v_{k}_biases']}
             for k in v_keys}

        return cls(learning_rate=learning_rate, t=int(data['t']), m=m, v=v, **kwargs)
