"""
Event Distance & Operational Surge Transformer
==============================================
Computes forward and backward temporal distances to promotional events and detects
reopening demand surges (pent-up customer demand).

Complexity:
- Time: O(N log N) via sorted GroupBy scans; vectorized forward/backward passes.
- Space: O(N) memory allocation for event indices.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer


class DayCounterTransformer(BaseFeatureTransformer):
    """
    Computes bidirectional day distances to promotions and detects post-closure demand spikes.
    """

    def __init__(self) -> None:
        self.is_fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "DayCounterTransformer":
        self.is_fitted_ = True
        return self

    @staticmethod
    def _compute_promo_distances(full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized bidirectional day distance counter for promotions.
        """
        full_df = full_df.sort_values(["HubID", "Date"]).copy()

        # Backward pass: DaysSinceLastPromo
        def calc_since(series: pd.Series) -> pd.Series:
            is_promo = series == 1
            idx = np.where(is_promo, np.arange(len(series)), np.nan)
            idx_ffill = pd.Series(idx).ffill().values
            return pd.Series(
                np.where(np.isnan(idx_ffill), 999.0, np.arange(len(series)) - idx_ffill),
                index=series.index,
            )

        # Forward pass: DaysUntilNextPromo
        def calc_until(series: pd.Series) -> pd.Series:
            is_promo = series == 1
            idx = np.where(is_promo, np.arange(len(series)), np.nan)
            idx_bfill = pd.Series(idx).bfill().values
            return pd.Series(
                np.where(np.isnan(idx_bfill), 999.0, idx_bfill - np.arange(len(series))),
                index=series.index,
            )

        full_df["DaysSinceLastPromo"] = (
            full_df.groupby("HubID")["PromoActive"].transform(calc_since).astype(np.float32)
        )
        full_df["DaysUntilNextPromo"] = (
            full_df.groupby("HubID")["PromoActive"].transform(calc_until).astype(np.float32)
        )

        return full_df

    @staticmethod
    def _compute_reopening_surge(full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects immediate hub reopening days following temporary refurbishment or closure.
        """
        full_df = full_df.sort_values(["HubID", "Date"]).copy()
        
        # Yesterday's operational status
        prev_open = full_df.groupby("HubID")["IsOpen"].shift(1).fillna(1)
        full_df["OpenAfterClosed"] = (
            (full_df["IsOpen"] == 1) & (prev_open == 0)
        ).astype(np.int8)

        # Count streak of preceding closed days
        def count_closed_streak(series: pd.Series) -> pd.Series:
            is_closed = (series == 0).astype(int)
            streak = is_closed.groupby((is_closed != is_closed.shift()).cumsum()).cumsum()
            return streak.shift(1).fillna(0)

        full_df["ClosedDaysStreak"] = (
            full_df.groupby("HubID")["IsOpen"].transform(count_closed_streak).astype(np.float32)
        )

        return full_df

    def transform_combined(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_tmp = train_df.copy()
        train_tmp["_partition_is_test"] = 0
        test_tmp = test_df.copy()
        test_tmp["_partition_is_test"] = 1

        combined = pd.concat([train_tmp, test_tmp], ignore_index=True)

        combined = self._compute_promo_distances(combined)
        combined = self._compute_reopening_surge(combined)

        train_out = (
            combined[combined["_partition_is_test"] == 0]
            .drop(columns=["_partition_is_test"])
            .reset_index(drop=True)
        )
        test_out = (
            combined[combined["_partition_is_test"] == 1]
            .drop(columns=["_partition_is_test"])
            .reset_index(drop=True)
        )

        return train_out, test_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._compute_reopening_surge(self._compute_promo_distances(df))
