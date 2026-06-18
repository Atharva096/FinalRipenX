from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionResponse(BaseModel):
    success: bool
    filename: str
    ripeness_class: str
    confidence: float
    confidence_percentage: dict
    harvest_estimate_days: Optional[int] = None
    cultivar: Optional[str] = None
    recommended_export_destination: Optional[str] = None
    export_logistics_action: Optional[str] = None
    mandatory_regulatory_compliance: Optional[list[str]] = None
    processed_at: datetime
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    detail: Optional[str] = None