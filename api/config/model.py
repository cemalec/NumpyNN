from pydantic import BaseModel, Field
from typing import Optional, List

class LayerConfig(BaseModel):
    type: str = Field(..., description="Type of the layer, e.g., 'Dense'")
    input_size: Optional[int] = Field(None, description="Input size for the layer")
    output_size: int = Field(..., description="Output size for the layer")
    activation_function: str = Field(..., description="Activation function to use, e.g., 'ReLU', 'Sigmoid'")
    name: Optional[str] = Field(None, description="Optional name for the layer")
    regularization: Optional[str] = Field(None, description="Regularization type, e.g., 'l1', 'l2'")
    lambda_reg: Optional[float] = Field(0.01, description="Regularization strength")

class ModelConfig(BaseModel):
    layers: List[LayerConfig] = Field(..., description="List of layer configurations")
    loss_function: str = Field(..., description="Loss function to use, e.g., 'CrossEntropy'")
    optimizer: str = Field(..., description="Optimizer to use, e.g., 'SGD', 'Adam'")
    learning_rate: float = Field(0.01, description="Learning rate for the optimizer")
    batch_size: int = Field(32, description="Batch size for training")
    epochs: int = Field(10, description="Number of epochs for training")
    model_name: Optional[str] = Field(None, description="Optional name for the model")