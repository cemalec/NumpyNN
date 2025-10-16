from pydantic import BaseModel, Field
from typing import Optional, List

class OptimizerConfig(BaseModel):
    type: str = Field(..., description="Type of the optimizer, e.g., 'SGD', 'Adam'")
    learning_rate: float = Field(0.01, description="Learning rate for the optimizer")
    beta1: Optional[float] = Field(0.9, description="Beta1 parameter for Adam optimizer")
    beta2: Optional[float] = Field(0.999, description="Beta2 parameter for Adam optimizer")
    epsilon: Optional[float] = Field(1e-8, description="Epsilon parameter for Adam optimizer")

class DatasetConfig(BaseModel):
    name: str = Field(..., description="Name of the dataset, e.g., 'MNIST', 'CIFAR10'")

class TrainingConfig(BaseModel):
    batch_size: int = Field(32, description="Batch size for training")
    epochs: int = Field(10, description="Number of epochs for training")
    learning_rate: float = Field(0.01, description="Learning rate for the optimizer")
    optimizer: OptimizerConfig = Field(..., description="Configuration for the optimizer")
    dataset: DatasetConfig = Field(..., description="Configuration for the dataset")
    model_save_path: Optional[str] = Field(None, description="Path to save the trained model")
    validation_split: float = Field(0.2, description="Fraction of training data to use for validation")
    shuffle: bool = Field(True, description="Whether to shuffle the dataset before each epoch")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

