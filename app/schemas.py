from pydantic import BaseModel, Field, field_validator
import datetime

class PredictionRequest(BaseModel):
    store: int = Field(..., description="Unique Store ID (1-1115)", ge=1, le=1115)
    date: datetime.date = Field(..., description="Transaction Date (YYYY-MM-DD)")
    promo: int = Field(..., description="Is promotion active (0 or 1)", ge=0, le=1)
    state_holiday: str = Field(..., description="State holiday indicator ('0', 'a', 'b', 'c' or 0, 1, 2, 3)")
    school_holiday: int = Field(..., description="Is school holiday active (0 or 1)", ge=0, le=1)

    @field_validator('state_holiday')
    def validate_state_holiday(cls, v):
        # Convert to string and strip
        val = str(v).strip()
        if val not in ['0', 'a', 'b', 'c', '1', '2', '3']:
            raise ValueError("state_holiday must be one of '0', 'a', 'b', 'c', '1', '2', '3'")
        return val

class PredictionResponse(BaseModel):
    predicted_sales: float = Field(..., description="Predicted demand sales value")
    confidence: str = Field(..., description="Prediction confidence level (High or Medium)")
    demand_level: str = Field(..., description="Categorical demand level (High, Normal, or Low)")
    inventory_recommendation: str = Field(..., description="Business recommendation for stock/inventory level")
    store_avg: float = Field(..., description="Historical average sales of the store")
    competition_distance: float = Field(..., description="Distance to nearest competitor in meters")
    store_type: str = Field(..., description="Type format of the store (A, B, C, or D)")

class ModelInfoResponse(BaseModel):
    name: str = Field(..., description="Name of the model algorithm")
    rmse: int = Field(..., description="Root Mean Squared Error metric achieved")
    r2: float = Field(..., description="R-squared metric achieved")
    trained: str = Field(..., description="Date the model was trained (YYYY-MM-DD)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the service")
