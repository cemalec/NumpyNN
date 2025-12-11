import logging
import yaml
from Layer import DenseLayer
from DifferentiableFunction import SoftMax, ReLU, CrossEntropyLoss
from Model import Model
from Optimizer import Adam, RMSProp,SGD
from typing import Any, Dict, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LayerConfig(BaseModel):
    type: str
    input_size: int
    output_size: int
    name: str = None
    activation_function: str = None

class CNNLayerConfig(BaseModel):
    type: str
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    name: str = None
    activation_function: str = None
    
class OptimizerConfig(BaseModel):
    type: str
    learning_rate: float = None
    beta1: float = None
    beta2: float = None
    epsilon: float = None

class LossConfig(BaseModel):
    type: str

class ModelConfig(BaseModel):
    layers: List[LayerConfig | CNNLayerConfig] = Field(...)
    loss: str = None
    optimizer: OptimizerConfig = None

def hydrate_model(filepath: str) -> Model:
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    # Convert to dataclass
    model_config = ModelConfig(
        layers=[LayerConfig(**layer) for layer in config["layers"]],
        loss=config["loss"],
        optimizer=OptimizerConfig(**config["optimizer"]))
    model = Model.from_dict(model_config.model_dump())
    return model