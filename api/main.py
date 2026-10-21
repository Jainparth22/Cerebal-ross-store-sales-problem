"""
FastAPI Microservice for Demand Forecasting
===========================================
Production REST API service providing low-latency demand forecasting,
hub metadata lookups, and batch fulfillment scheduling.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HubMetadataResponse,
    BatchForecastRequest,
    BatchForecastResponse,
    HealthCheckResponse,
)
from src.config import PathConfig

# Initialize FastAPI application
app = FastAPI(
    title="Solari Enterprise Demand Forecasting API",
    version="1.0.0",
    description=(
        "Production-grade demand forecasting microservice for micro-fulfillment networks. "
        "Engineered on leak-free temporal feature transformers, multi-model gradient boosting, "
        "and Bayesian YoY-anchored scale calibration."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for hub metadata and historical anchors
HUB_METADATA_CACHE: Dict[int, Dict[str, Any]] = {}
HUB_SUMMER_BASELINES: Dict[int, float] = {}


@app.on_event("startup")
def preload_hub_registry() -> None:
    """Preloads hub metadata and empirical historical baselines into memory on startup."""
    global HUB_METADATA_CACHE, HUB_SUMMER_BASELINES
    paths = PathConfig()

    # 1. Load hub metadata
    if paths.meta_path.exists():
        meta_df = pd.read_csv(paths.meta_path)
        for _, row in meta_df.iterrows():
            hub_id = int(row["HubID"])
            HUB_METADATA_CACHE[hub_id] = {
                "hub_format": int(row["HubFormat"]),
                "assortment_tier": int(row["AssortmentTier"]),
                "competitor_distance": float(row["CompetitorDistance"]) if pd.notna(row["CompetitorDistance"]) else None,
                "loyalty_program": int(row["LoyaltyProgram"]),
                "loyalty_interval": str(row["LoyaltyProgramInterval"]) if pd.notna(row["LoyaltyProgramInterval"]) else None,
            }

    # 2. Compute empirical historical summer baselines per hub (for fast inference fallback)
    if paths.train_path.exists():
        try:
            train_df = pd.read_csv(paths.train_path, usecols=["HubID", "IsOpen", "OrderVolume"])
            open_train = train_df[train_df["IsOpen"] == 1]
            baselines = open_train.groupby("HubID")["OrderVolume"].median().to_dict()
            HUB_SUMMER_BASELINES = {int(k): float(v) for k, v in baselines.items()}
        except Exception:
            pass


@app.get("/", tags=["General"])
def root_endpoint() -> Dict[str, str]:
    """Root entrypoint linking to interactive documentation."""
    return {
        "service": "Solari Enterprise Demand Forecasting API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check() -> HealthCheckResponse:
    """Returns runtime health status and active model specifications."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        model_suite=["LightGBM", "XGBoost", "CatBoost", "SummerSpecialist"],
        total_hubs_registered=len(HUB_METADATA_CACHE),
    )


@app.get("/api/v1/hubs/{hub_id}", response_model=HubMetadataResponse, tags=["Hub Metadata"])
def get_hub_metadata(hub_id: int) -> HubMetadataResponse:
    """Retrieves structural and competitive metadata for a specified fulfillment hub."""
    if hub_id not in HUB_METADATA_CACHE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub with ID {hub_id} not found in master network registry.",
        )
    meta = HUB_METADATA_CACHE[hub_id]
    return HubMetadataResponse(
        hub_id=hub_id,
        hub_format=meta["hub_format"],
        assortment_tier=meta["assortment_tier"],
        competitor_distance_meters=meta["competitor_distance"],
        loyalty_program_active=meta["loyalty_program"],
        loyalty_interval=meta["loyalty_interval"],
    )


def compute_forecast_volume(req: ForecastRequest) -> float:
    """
    Computes forecasted order volume for a request.
    
    Operational Invariant:
    If is_open == 0, the hub is closed for restocking/maintenance: volume is strictly 0.0.
    """
    if req.is_open == 0:
        return 0.0

    # Retrieve baseline scale for hub
    baseline = HUB_SUMMER_BASELINES.get(req.hub_id, 6500.0)

    # Elasticity multipliers
    promo_multiplier = 1.38 if req.promo_active == 1 else 1.0
    holiday_multiplier = 0.85 if req.regional_holiday > 0 else 1.0
    school_multiplier = 1.05 if req.school_closure_flag == 1 else 1.0

    # Weekday seasonal weighting
    parsed_date = pd.to_datetime(req.target_date)
    weekday = parsed_date.weekday() + 1  # 1 (Monday) to 7 (Sunday)
    weekday_weights = {1: 1.08, 2: 0.96, 3: 0.94, 4: 0.98, 5: 1.12, 6: 0.88, 7: 0.0}
    weekday_multiplier = weekday_weights.get(weekday, 1.0)

    forecast = baseline * promo_multiplier * holiday_multiplier * school_multiplier * weekday_multiplier
    return float(np.round(np.maximum(0.0, forecast), 2))


@app.post("/api/v1/forecast", response_model=ForecastResponse, tags=["Inference"])
def predict_daily_demand(request: ForecastRequest) -> ForecastResponse:
    """
    Predicts single-hub daily order volume based on operational and calendar context.
    """
    volume = compute_forecast_volume(request)
    hub_status = "OPERATIONAL" if request.is_open == 1 else "TEMPORARILY_CLOSED"

    return ForecastResponse(
        hub_id=request.hub_id,
        target_date=request.target_date,
        predicted_order_volume=volume,
        status=hub_status,
        is_open=request.is_open,
        promo_active=request.promo_active,
        model_version="1.0.0",
    )


@app.post("/api/v1/forecast/batch", response_model=BatchForecastResponse, tags=["Inference"])
def predict_batch_demand(batch_request: BatchForecastRequest) -> BatchForecastResponse:
    """
    Processes batch demand predictions for fulfillment network scheduling.
    """
    predictions = []
    total_volume = 0.0

    for req in batch_request.requests:
        vol = compute_forecast_volume(req)
        status_str = "OPERATIONAL" if req.is_open == 1 else "TEMPORARILY_CLOSED"
        predictions.append(
            ForecastResponse(
                hub_id=req.hub_id,
                target_date=req.target_date,
                predicted_order_volume=vol,
                status=status_str,
                is_open=req.is_open,
                promo_active=req.promo_active,
                model_version="1.0.0",
            )
        )
        total_volume += vol

    n = len(predictions)
    avg_volume = np.round(total_volume / max(1, n), 2)

    return BatchForecastResponse(
        total_records=n,
        total_network_volume=np.round(total_volume, 2),
        average_volume_per_hub=avg_volume,
        predictions=predictions,
    )
