import logging
import yaml
from typing import List,Tuple,TypeAlias,Optional
from Layer import DenseLayer
from DifferentiableFunction import SoftMax, ReLU, CrossEntropyLoss
from Model import Model
from Optimizer import Adam, RMSProp,SGD
from typing import Any, Dict, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LayerConfig(BaseModel):
    type: str = "DenseLayer"
    input_size: int
    output_size: int
    name: str = None
    activation_function: str = None

class CNNLayerConfig(LayerConfig):
    type: str = "CNNLayer"
    input_size: Tuple[int, int, int]
    output_size: Tuple[int, int, int]
    kernel_size: int | List[int]
    num_filters: int
    stride: int = 1
    padding: int = 0

class FlattenLayerConfig(BaseModel):
    type: str = "FlattenLayer"

class ReshapeLayerConfig(BaseModel):
    type: str = "ReshapeLayer"
    output_shape: Tuple[int, ...]
    name: str = None

class OptimizerConfig(BaseModel):
    type: str
    learning_rate: float = None
    beta1: float = None
    beta2: float = None
    epsilon: float = None

class LossConfig(BaseModel):
    type: str

LayerConfigType: TypeAlias = LayerConfig | CNNLayerConfig | FlattenLayerConfig | ReshapeLayerConfig
class ModelConfig(BaseModel):
    layers: List[LayerConfigType] = Field(...)
    loss: str = None
    optimizer: OptimizerConfig = None

def get_layer_model(layer_type: str) -> LayerConfigType:
    logger.debug(f"Getting layer model for type: {layer_type}")
    if layer_type == "DenseLayer":
        return LayerConfig
    elif layer_type == "CNNLayer":
        return CNNLayerConfig
    elif layer_type == "FlattenLayer":
        return FlattenLayerConfig
    elif layer_type == "ReshapeLayer":
        return ReshapeLayerConfig
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    
def hydrate_model(filepath: str) -> Model:
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    # Convert to dataclass
    model_config = ModelConfig(
        layers=[get_layer_model(layer_config["type"])(**layer_config) for layer_config in config["layers"]],
        loss=config["loss"],
        optimizer=OptimizerConfig(**config["optimizer"]))
    model = Model.from_dict(model_config.model_dump())
    return model