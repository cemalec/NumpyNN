from pydantic import BaseModel, Field
from typing import Optional, List

class TrainingRequest(BaseModel):
    model_config_path: str = Field(..., description="Path to the model configuration file")
    training_config_path: str = Field(..., description="Path to the training configuration file")
    evaluation_config_path: Optional[str] = Field(None, description="Path to the evaluation configuration file, if any")
    dataset_path: Optional[str] = Field(None, description="Path to the dataset, if not specified in training config")
    output_model_path: Optional[str] = Field(None, description="Path to save the trained model")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class PredictionRequest(BaseModel):
    model_path: str = Field(..., description="Path to the trained model file")
    input_data: List[List[float]] = Field(..., description="Input data for prediction as a list of feature lists")
    batch_size: Optional[int] = Field(32, description="Batch size for making predictions")
    output_path: Optional[str] = Field(None, description="Path to save the predictions, if any")