from pydantic import BaseModel, Field
from typing import Optional

class TrainingResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the training was successful")
    model_path: Optional[str] = Field(None, description="Path where the trained model is saved")
    epochs_completed: Optional[int] = Field(None, description="Number of epochs completed during training")
    final_loss: Optional[float] = Field(None, description="Final loss value after training")
    message: Optional[str] = Field(None, description="Additional information or error message")

class PredictionResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the prediction was successful")
    predictions: Optional[list] = Field(None, description="List of predictions made by the model")
    output_path: Optional[str] = Field(None, description="Path where the predictions are saved, if applicable")
    message: Optional[str] = Field(None, description="Additional information or error message")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Indicates if the operation was successful")
    error_code: int = Field(..., description="Error code representing the type of error")
    error_message: str = Field(..., description="Detailed error message")