from abc import abstractmethod
from typing import Dict, Any
import numpy as np


class Optimizer:
    def __init__(self):
        self.name = None
        self.type = "Optimizer"

    @abstractmethod
    def step(
        self, layer: Any, parameter_gradients: Dict[str, np.ndarray]
    ) -> np.ndarray:
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

    def step(self, layer, parameter_gradients: Dict[str, np.ndarray]) -> np.ndarray:
        for param_name, gradient in parameter_gradients.items():
            if gradient is None:
                continue

            if not hasattr(layer, param_name):
                continue

            param_val = getattr(layer, param_name)
            if param_val is not None:
                setattr(layer, param_name, param_val - self.learning_rate * gradient)

    def to_dict(self) -> dict:
        return {"learning_rate": self.learning_rate, "type": self.type}

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
        self.s[layer.name] = {}
        for attr_name in ["weights", "biases", "gamma", "beta"]:
            if hasattr(layer, attr_name):
                param = getattr(layer, attr_name)
                if param is not None:
                    self.s[layer.name][attr_name] = np.zeros_like(param)

    def step(
        self, layer: Any, parameter_gradients: Dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.s.get(layer.name) is None:
            self.initialize_state(layer)

        for param_name, gradient in parameter_gradients.items():
            if gradient is None:
                continue

            if not hasattr(layer, param_name):
                continue

            param_val = getattr(layer, param_name)
            if param_val is None:
                continue

            if param_name not in self.s[layer.name]:
                self.s[layer.name][param_name] = np.zeros_like(gradient)

            self.s[layer.name][param_name] = self.beta * self.s[layer.name][
                param_name
            ] + (1 - self.beta) * (gradient**2)

            update = (
                self.learning_rate
                * gradient
                / (np.sqrt(self.s[layer.name][param_name]) + self.epsilon)
            )
            setattr(layer, param_name, param_val - update)

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "type": self.type,
        }

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
        self.m = m if m is not None else dict()
        self.v = v if v is not None else dict()
        self.t = t

    def initialize_state(self, layer: Any):
        """Initialize momentum and velocity for all learnable parameters."""
        self.m[layer.name] = {}
        self.v[layer.name] = {}

        # Find all learnable parameters (weights, biases, gamma, beta, etc.)
        for attr_name in ["weights", "biases", "gamma", "beta"]:
            if hasattr(layer, attr_name):
                param = getattr(layer, attr_name)
                if param is not None:
                    self.m[layer.name][attr_name] = np.zeros_like(param)
                    self.v[layer.name][attr_name] = np.zeros_like(param)

    def step(
        self, layer: Any, parameter_gradients: Dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.m.get(layer.name) is None:
            self.initialize_state(layer)

        self.t += 1

        # Process all gradient keys generically
        for param_name, gradient in parameter_gradients.items():
            if gradient is None:
                continue

            # Check if layer has this parameter and it's learnable
            if not hasattr(layer, param_name):
                continue

            param_val = getattr(layer, param_name)
            if param_val is None:
                continue

            # Initialize momentum/velocity if needed
            if param_name not in self.m[layer.name]:
                self.m[layer.name][param_name] = np.zeros_like(gradient)
                self.v[layer.name][param_name] = np.zeros_like(gradient)

            # Adam update
            self.m[layer.name][param_name] = (
                self.beta1 * self.m[layer.name][param_name]
                + (1 - self.beta1) * gradient
            )
            self.v[layer.name][param_name] = self.beta2 * self.v[layer.name][
                param_name
            ] + (1 - self.beta2) * (gradient**2)

            m_hat = self.m[layer.name][param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[layer.name][param_name] / (1 - self.beta2**self.t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(layer, param_name, param_val - update)

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "type": self.type,
            "m": self.m,
            "v": self.v,
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            learning_rate=data.get("learning_rate", 0.001),
            beta1=data.get("beta1", 0.9),
            beta2=data.get("beta2", 0.999),
            epsilon=data.get("epsilon", 1e-8),
            m=data.get("m"),
            v=data.get("v"),
            t=data.get("t", 0),
        )

    def save_state(self, filepath: str):
        """Save optimizer state to npz file."""
        np.savez(
            filepath,
            t=self.t,
            m=np.array(self.m, dtype=object),
            v=np.array(self.v, dtype=object),
        )

    @classmethod
    def load_state(cls, filepath: str, learning_rate: float, **kwargs):
        """Load optimizer state from npz file."""
        data = np.load(filepath, allow_pickle=True)
        m = data["m"].item()
        v = data["v"].item()

        return cls(learning_rate=learning_rate, t=int(data["t"]), m=m, v=v, **kwargs)
