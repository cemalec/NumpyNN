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

    def step(self, layer, grads: Dict[str, np.ndarray]) -> np.ndarray:
        for param_name, grad in grads.items():
            if grad is None or param_name == "inputs":
                continue
            
            if not hasattr(layer, param_name):
                continue
            
            param_val = getattr(layer, param_name)
            if param_val is not None:
                setattr(layer, param_name, param_val - self.learning_rate * grad)

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
        for attr_name in ['weights', 'biases', 'gamma', 'beta']:
            if hasattr(layer, attr_name):
                param = getattr(layer, attr_name)
                if param is not None:
                    self.s[layer.name][attr_name] = np.zeros_like(param)

    def step(self, layer: Any, grads: Dict[str, np.ndarray]) -> np.ndarray:
        if self.s.get(layer.name) is None:
            self.initialize_state(layer)
        
        for param_name, grad in grads.items():
            if grad is None or param_name == "inputs":
                continue
            
            if not hasattr(layer, param_name):
                continue
            
            param_val = getattr(layer, param_name)
            if param_val is None:
                continue
            
            if param_name not in self.s[layer.name]:
                self.s[layer.name][param_name] = np.zeros_like(grad)
            
            self.s[layer.name][param_name] = self.beta * self.s[layer.name][param_name] + (
                1 - self.beta
            ) * (grad ** 2)
            
            update = (
                self.learning_rate * grad / (np.sqrt(self.s[layer.name][param_name]) + self.epsilon)
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
        for attr_name in ['weights', 'biases', 'gamma', 'beta']:
            if hasattr(layer, attr_name):
                param = getattr(layer, attr_name)
                if param is not None:
                    self.m[layer.name][attr_name] = np.zeros_like(param)
                    self.v[layer.name][attr_name] = np.zeros_like(param)

    def step(self, layer: Any, grads: Dict[str, np.ndarray]) -> np.ndarray:
        if self.m.get(layer.name) is None:
            self.initialize_state(layer)
        
        self.t += 1
        
        # Process all gradient keys generically
        for param_name, grad in grads.items():
            if grad is None or param_name == "inputs":
                continue
            
            # Check if layer has this parameter and it's learnable
            if not hasattr(layer, param_name):
                continue
            
            param_val = getattr(layer, param_name)
            if param_val is None:
                continue
            
            # Initialize momentum/velocity if needed
            if param_name not in self.m[layer.name]:
                self.m[layer.name][param_name] = np.zeros_like(grad)
                self.v[layer.name][param_name] = np.zeros_like(grad)
            
            # Adam update
            self.m[layer.name][param_name] = (
                self.beta1 * self.m[layer.name][param_name] + (1 - self.beta1) * grad
            )
            self.v[layer.name][param_name] = (
                self.beta2 * self.v[layer.name][param_name] + (1 - self.beta2) * (grad ** 2)
            )
            
            m_hat = self.m[layer.name][param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer.name][param_name] / (1 - self.beta2 ** self.t)
            
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(layer, param_name, param_val - update)

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
        return cls(
            learning_rate=data.get("learning_rate", 0.001),
            beta1=data.get("beta1", 0.9),
            beta2=data.get("beta2", 0.999),
            epsilon=data.get("epsilon", 1e-8),
        )

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
