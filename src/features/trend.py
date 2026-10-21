"""
Per-Hub Trend Estimation Transformer (Ridge Regression)
======================================================
Fits an isolated regularized linear model per hub on the trailing 365-day training
window to isolate macro growth/decay trajectories from day-of-week and promo noise.

Reference:
Gert Jacobusse (Rossmann 1st Place): "fit a store specific linear model on 1) day number
2) day of week 3) promotions — use predicted slope as feature"
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features.base import BaseFeatureTransformer


class HubTrendTransformer(BaseFeatureTransformer):
    """
    Computes per-hub regression slope on temporal index to extrapolate structural growth.
    
    Mathematical / CS Rationale:
    ----------------------------
    Raw rolling averages confound cyclical weekly patterns with true secular growth trends.
    By fitting an L2-regularized linear model (Ridge Regression) with covariates for
    Weekday and Promotion activity:
        OrderVolume = beta_0 + beta_1 * DayNum + beta_2 * Weekday + beta_3 * Promo
    we isolate beta_1 as the uncontaminated linear trend parameter via:
        beta = (X^T X + alpha * I)^(-1) X^T y
    With p=3 features and T <= 365 observations per hub, closed-form inversion is O(p^3) = O(27),
    yielding an ultra-fast O(H * T) aggregate runtime for H=1,115 hubs.
    """

    def __init__(self, lookback_days: int = 365, alpha: float = 1.0) -> None:
        self.lookback_days = lookback_days
        self.alpha = alpha
        self.slopes_: Optional[pd.DataFrame] = None
        self.global_slope_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HubTrendTransformer":
        cutoff = train_df["Date"].max()
        lookback_start = cutoff - pd.Timedelta(days=self.lookback_days)

        recent = train_df[
            (train_df["Date"] >= lookback_start) & (train_df["IsOpen"] == 1)
        ].copy()

        min_date = recent["Date"].min()
        recent["day_num"] = (recent["Date"] - min_date).dt.days

        slopes: Dict[int, float] = {}
        for hub_id, grp in recent.groupby("HubID"):
            if len(grp) >= 30:  # Require sufficient sample degrees of freedom
                X = grp[["day_num", "Weekday", "PromoActive"]].values
                y = grp["OrderVolume"].values
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(X, y)
                slopes[int(hub_id)] = float(model.coef_[0])
            else:
                slopes[int(hub_id)] = 0.0

        self.slopes_ = pd.DataFrame(
            list(slopes.items()), columns=["HubID", "hub_trend_slope"]
        )
        self.global_slope_ = float(self.slopes_["hub_trend_slope"].median())

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.slopes_ is not None, "Transformer must be fitted before transform"
        df = df.merge(self.slopes_, on="HubID", how="left")
        df["hub_trend_slope"] = df["hub_trend_slope"].fillna(self.global_slope_).astype(np.float32)
        return df
