"""
Annual Cyclic Lag & Rolling Statistics Transformer
==================================================
Computes year-over-year cyclic lags and causal rolling statistics with strict leakage guards.

Mathematical Rationale:
-----------------------
Demand in retail/quick-commerce exhibits strong dual periodicity:
1. Intra-week periodicity: period T_w = 7 days.
2. Annual seasonality: period T_a = 365.25 days.
A 365-day lag introduces a 1-day day-of-week phase shift (e.g. comparing Sunday to Saturday).
The lag of 364 days (52 * 7) preserves BOTH annual seasonality and phase-locked day-of-week
alignment:
    t - 364 = t - (52 * 7)  ==> Weekday(t) == Weekday(t - 364)
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer


class LagRollingTransformer(BaseFeatureTransformer):
    """
    Constructs annual cyclic lags and causal rolling statistics.
    """

    def __init__(
        self,
        lag_days: List[int] = [7, 14, 21, 28, 364, 365, 371, 728],
        rolling_windows: List[int] = [7, 14, 28],
    ) -> None:
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows

    def fit(self, train_df: pd.DataFrame) -> "LagRollingTransformer":
        return self

    def transform_combined(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates lags across joint time axis and applies YoY fallback for test partition.
        """
        train_tmp = train_df.copy()
        train_tmp["_partition_is_test"] = 0
        test_tmp = test_df.copy()
        if "OrderVolume" not in test_tmp.columns:
            test_tmp["OrderVolume"] = np.nan
        test_tmp["_partition_is_test"] = 1

        full = pd.concat([train_tmp, test_tmp], ignore_index=True)
        full = full.sort_values(["HubID", "Date"]).copy()

        # Compute point lags
        for lag in self.lag_days:
            full[f"vol_lag_{lag}"] = (
                full.groupby("HubID")["OrderVolume"].shift(lag).astype(np.float32)
            )

        # Compute causal rolling statistics (strictly shift 1 before windowing)
        shifted = full.groupby("HubID")["OrderVolume"].shift(1)
        for w in self.rolling_windows:
            full[f"vol_roll{w}_mean"] = (
                shifted.groupby(full["HubID"])
                .rolling(w, min_periods=max(2, w // 2))
                .mean()
                .reset_index(level=0, drop=True)
                .astype(np.float32)
            )

        full["vol_roll7_std"] = (
            shifted.groupby(full["HubID"])
            .rolling(7, min_periods=3)
            .std()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        if "vol_roll7_mean" in full.columns and "vol_roll28_mean" in full.columns:
            full["vol_roll7_dev"] = (
                full["vol_roll7_mean"] - full["vol_roll28_mean"]
            ).astype(np.float32)

        train_out = (
            full[full["_partition_is_test"] == 0]
            .drop(columns=["_partition_is_test"])
            .reset_index(drop=True)
        )
        test_out = (
            full[full["_partition_is_test"] == 1]
            .drop(columns=["_partition_is_test"])
            .reset_index(drop=True)
        )

        # Test set imputation: fallback to YoY anchors if available
        if not test_out.empty:
            for lag in [7, 14, 21, 28]:
                col = f"vol_lag_{lag}"
                if col in test_out.columns:
                    if "vol_lag_364" in test_out.columns:
                        test_out[col] = test_out[col].fillna(test_out["vol_lag_364"])
                    if "vol_lag_365" in test_out.columns:
                        test_out[col] = test_out[col].fillna(test_out["vol_lag_365"])

            if "vol_roll7_mean" in test_out.columns and "vol_lag_364" in test_out.columns:
                test_out["vol_roll7_mean"] = test_out["vol_roll7_mean"].fillna(test_out["vol_lag_364"])
            if "vol_roll14_mean" in test_out.columns and "vol_lag_365" in test_out.columns:
                test_out["vol_roll14_mean"] = test_out["vol_roll14_mean"].fillna(test_out["vol_lag_365"])
            if "vol_roll28_mean" in test_out.columns and "vol_lag_371" in test_out.columns:
                test_out["vol_roll28_mean"] = test_out["vol_roll28_mean"].fillna(test_out["vol_lag_371"])

        return train_out, test_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["HubID", "Date"]).copy()
        for lag in self.lag_days:
            df[f"vol_lag_{lag}"] = (
                df.groupby("HubID")["OrderVolume"].shift(lag).astype(np.float32)
            )
        return df
