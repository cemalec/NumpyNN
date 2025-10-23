from abc import abstractmethod
from typing import Dict,Any
import numpy as np

class Optimizer:
    def __init__(self):
        self.name = self.__class__.__name__
        self.type = 'Optimizer'
    @abstractmethod
    def step(self, layer: Any, grads: Dict[str,np.ndarray]) -> np.ndarray:
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str,Any]:
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str,Any]) -> 'Optimizer':
        pass
    
class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.type = 'SGD'
    
    def step(self, layer, grads: np.ndarray) -> np.ndarray:
        params={'weights': layer.weights,'biases': layer.biases} 
        for key in ['weights','biases']:
            params[key] -= self.learning_rate * grads[key]
        return params['weights'], params['biases']
    def to_dict(self) -> dict:
        return {'learning_rate': self.learning_rate}
    @classmethod
    def from_dict(cls, data: dict):
        return cls(learning_rate=data['learning_rate'])

class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__()
        self.type = 'RMSProp'
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = dict()
    
    def initialize_state(self,layer: Any):
        self.s[layer.name] = dict()
        self.s[layer.name]['weights'] = np.zeros_like(layer.weights)
        self.s[layer.name]['biases'] = np.zeros_like(layer.biases)

    def step(self,
             layer: Any,
             grads: Dict[str,np.ndarray]) -> np.ndarray:
        if self.s.get(layer.name) is None:
            self.initialize_state(layer)
        params = {'weights': layer.weights, 'biases': layer.biases}
        for key in ['weights', 'biases']:
            self.s[layer.name][key] = self.beta * self.s[layer.name][key] + (1 - self.beta) * (grads[key] ** 2)
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.s[layer.name][key]) + self.epsilon)
        return params['weights'], params['biases']
    
    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({'learning_rate': self.learning_rate,
                          'beta': self.beta, 
                          'epsilon': self.epsilon,
                          'type': self.type})
        return base_dict
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(learning_rate=data['learning_rate'],
                   beta=data.get('beta', 0.9),
                   epsilon=data.get('epsilon', 1e-8))
     
class Adam(Optimizer):
    def __init__(self, 
                 learning_rate: float, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8):
        super().__init__()
        self.type = 'Adam'
        self.learning_rate = learning_rate
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
        if self.t < 1000:
            self.t += 1
        for key in ['weights', 'biases']:
            self.m[layer.name][key] = self.beta1 * self.m[layer.name][key]  + (1 - self.beta1) * grads[key]
            self.v[layer.name][key]  = self.beta2 * self.v[layer.name][key]  + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[layer.name][key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer.name][key] / (1 - self.beta2 ** self.t)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params['weights'], params['biases']
    
    def to_dict(self) -> dict:
        return {'beta1': self.beta1, 
                          'beta2': self.beta2, 
                          'epsilon': self.epsilon,
                          'm': self.m,
                          'v': self.v,
                          't': self.t}
    
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