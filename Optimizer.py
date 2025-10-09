from abc import abstractmethod
from typing import Dict,Any
import numpy as np

class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def step(self, layer: Any, grads: Dict[str,np.ndarray]) -> np.ndarray:
        pass

    def to_dict(self) -> dict:
        return {'learning_rate': self.learning_rate}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
class SGD(Optimizer):
    def step(self, layer, grads: np.ndarray) -> np.ndarray:
        params={'weights': layer.weights,'biases': layer.biases} 
        for key in ['weights','biases']:
            params[key] -= self.learning_rate * grads[key]
        return params['weights'], params['biases']
    
class Adam(Optimizer):
    def __init__(self, 
                 learning_rate: float, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()
        self.t = 0
    
    def initialize_state(self,layer: Any):
        self.m[layer.name] = dict()
        self.v[layer.name] = dict()
        self.m[layer.name]['weights'] = np.zeros_like(layer.weights)
        self.v[layer.name]['weights'] = np.zeros_like(layer.weights)
        self.m[layer.name]['biases'] = np.zeros_like(layer.biases)
        self.v[layer.name]['biases'] = np.zeros_like(layer.biases)

    def step(self,
             layer: Any,
             grads: Dict[str,np.ndarray]) -> np.ndarray:
        if self.m.get(layer.name) is None:
            self.initialize_state(layer)
        params = {'weights': layer.weights, 'biases': layer.biases}
        self.t += 1
        for key in ['weights', 'biases']:
            self.m[layer.name][key] = self.beta1 * self.m[layer.name][key]  + (1 - self.beta1) * grads[key]
            self.v[layer.name][key]  = self.beta2 * self.v[layer.name][key]  + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[layer.name][key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer.name][key] / (1 - self.beta2 ** self.t)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params['weights'], params['biases']
    
    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({'beta1': self.beta1, 
                          'beta2': self.beta2, 
                          'epsilon': self.epsilon,
                          'm': self.m,
                          'v': self.v,
                          't': self.t})
        return base_dict
    
    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(learning_rate=data['learning_rate'],
                  beta1=data.get('beta1', 0.9),
                  beta2=data.get('beta2', 0.999),
                  epsilon=data.get('epsilon', 1e-8))
        obj.m = data.get('m', None)
        obj.v = data.get('v', None)
        obj.t = data.get('t', 0)
        return obj