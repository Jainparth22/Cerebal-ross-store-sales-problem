"""
Summer Specialist Forecaster
============================
Domain specialist LightGBM model trained exclusively on May through September observations.

Domain Rationale:
-----------------
The 42-day evaluation window spans late June through July (summer seasonality).
While annual full-history models capture broad macro patterns, consumer purchasing behavior
shifts fundamentally in summer (school closures, regional holiday spikes, heat dynamics).
Training a dedicated model on summer months across all available years (2013-2015) provides
an uncorrelated, seasonally calibrated ensemble component.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.models.lgbm_model import LightGBMForecaster
from src.config import ModelParamsConfig


class SummerSpecialistForecaster(LightGBMForecaster):
    """
    Sub-window specialist filtering training partition to May-September rows.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: int = 150,
        verbose: bool = False,
    ) -> "SummerSpecialistForecaster":
        assert "Month" in X_train.columns, "Month feature required for SummerSpecialist"
        
        # Filter training data to May-September (months 5-9)
        summer_mask = X_train["Month"].isin([5, 6, 7, 8, 9])
        X_sub = X_train[summer_mask].reset_index(drop=True)
        y_sub = y_train[summer_mask]

        return super().fit(
            X_sub,
            y_sub,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
