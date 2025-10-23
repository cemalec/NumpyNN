import yaml
from Layer import DenseLayer
from DifferentiableFunction import SoftMax, ReLU,CrossEntropyLoss
from Model import Model
from Optimizer import Adam
from typing import Any, Dict

basic_model = Model(
    layers=[DenseLayer(input_size=784,
                        output_size=64,
                        activation_function=ReLU()), 
            DenseLayer(input_size=32,
                        output_size=10,
                        activation_function=SoftMax())],
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=1e-3)
)
print(basic_model.to_dict())
def hydrate_model(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)