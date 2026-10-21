"""
API Schema Specifications (Pydantic Models)
===========================================
Defines strictly typed request and response contracts for demand forecasting endpoints.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import date


class ForecastRequest(BaseModel):
    """Payload contract for single-hub daily demand inference."""
    hub_id: int = Field(..., ge=1, le=1115, description="Unique Hub ID (1 to 1,115)")
    target_date: str = Field(..., description="Target forecast date in YYYY-MM-DD format")
    is_open: int = Field(1, ge=0, le=1, description="Operational status: 1 = Open, 0 = Closed")
    promo_active: int = Field(0, ge=0, le=1, description="Promotion flag: 1 = Active, 0 = Inactive")
    regional_holiday: int = Field(0, ge=0, le=3, description="Holiday flag (0=None, 1=Public, 2=Religious, 3=Winter)")
    school_closure_flag: int = Field(0, ge=0, le=1, description="School closure indicator (1 = Closed, 0 = Open)")

    @field_validator("target_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("target_date must follow ISO 8601 format (YYYY-MM-DD)")


class ForecastResponse(BaseModel):
    """Output contract containing predicted volume and operational metadata."""
    hub_id: int
    target_date: str
    predicted_order_volume: float = Field(..., ge=0.0, description="Predicted daily order total")
    status: str = Field(..., description="Hub status: 'OPERATIONAL' or 'TEMPORARILY_CLOSED'")
    is_open: int
    promo_active: int
    model_version: str = "1.0.0"


class HubMetadataResponse(BaseModel):
    """Hub static master record specifications."""
    hub_id: int
    hub_format: int
    assortment_tier: int
    competitor_distance_meters: Optional[float]
    loyalty_program_active: int
    loyalty_interval: Optional[str]


class BatchForecastRequest(BaseModel):
    """Payload contract for multi-hub or multi-day batch predictions."""
    requests: List[ForecastRequest]


class BatchForecastResponse(BaseModel):
    """Aggregated output contract for batch execution."""
    total_records: int
    total_network_volume: float
    average_volume_per_hub: float
    predictions: List[ForecastResponse]


class HealthCheckResponse(BaseModel):
    """System health check and runtime status contract."""
    status: str
    version: str
    model_suite: List[str]
    total_hubs_registered: int
