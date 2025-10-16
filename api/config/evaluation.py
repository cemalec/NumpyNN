from pydantic import BaseModel, Field
from typing import Optional, List

class MetricConfig(BaseModel):
    name: str = Field(..., description="Name of the metric, e.g., 'accuracy', 'precision', 'recall'")
    threshold: Optional[float] = Field(None, description="Threshold for binary classification metrics")
    average: Optional[str] = Field(None, description="Type of averaging for multi-class metrics, e.g., 'macro', 'micro'")

class EvaluationConfig(BaseModel):
    metrics: List[MetricConfig] = Field(..., description="List of metrics to evaluate the model")
    validation_split: float = Field(0.2, description="Fraction of training data to use for validation")
    batch_size: int = Field(32, description="Batch size for evaluation")
    shuffle: bool = Field(False, description="Whether to shuffle the dataset before evaluation")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility during evaluation")

