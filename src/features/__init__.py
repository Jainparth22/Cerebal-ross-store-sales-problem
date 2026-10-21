"""
Feature Engineering Package
===========================
Modular feature transformers implementing the Scikit-Learn transformer protocol.
"""

from src.features.base import BaseFeatureTransformer
from src.features.ratios import HubRatioTransformer
from src.features.trend import HubTrendTransformer
from src.features.distance import DayCounterTransformer
from src.features.lags import LagRollingTransformer
from src.features.target_encoder import OutOfFoldTargetEncoder
from src.features.pipeline import FeaturePipeline

__all__ = [
    "BaseFeatureTransformer",
    "HubRatioTransformer",
    "HubTrendTransformer",
    "DayCounterTransformer",
    "LagRollingTransformer",
    "OutOfFoldTargetEncoder",
    "FeaturePipeline",
]
