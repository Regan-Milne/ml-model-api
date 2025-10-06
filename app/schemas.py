from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PredictionInput(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOutput(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionInput(BaseModel):
    instances: List[PredictionInput]
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    },
                    {
                        "sepal_length": 6.7,
                        "sepal_width": 3.1,
                        "petal_length": 4.7,
                        "petal_width": 1.5
                    }
                ]
            }
        }

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
