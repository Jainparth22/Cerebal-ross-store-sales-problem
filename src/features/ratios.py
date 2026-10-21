"""
Hub Ratio & Elasticity Transformer
==================================
Computes per-hub static behavioral aggregates reflecting customer elasticity to
promotions, regional holidays, and weekend demand cycles.

Reference:
Gert Jacobusse (Rossmann 1st Place): "aggregates by store: ratio of sales during
promotions/holidays/Saturdays, proportion school holidays, proportion of days open"
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer


class HubRatioTransformer(BaseFeatureTransformer):
    """
    Extracts hub-level static elasticity features from historical open-day records.
    """

    def __init__(self) -> None:
        self.hub_ratios_: Optional[pd.DataFrame] = None
        self.global_defaults_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "HubRatioTransformer":
        open_df = train_df[train_df["IsOpen"] == 1].copy()

        # Overall open-day mean
        mean_vol = open_df.groupby("HubID")["OrderVolume"].mean().rename("hub_mean_vol")

        # Promo elasticity ratio
        promo_mean = open_df[open_df["PromoActive"] == 1].groupby("HubID")["OrderVolume"].mean()
        non_promo_mean = open_df[open_df["PromoActive"] == 0].groupby("HubID")["OrderVolume"].mean()
        promo_ratio = (promo_mean / non_promo_mean.clip(lower=1.0)).rename("promo_sales_ratio")

        # Holiday elasticity ratio
        hol_mean = open_df[open_df["RegionalHoliday"] > 0].groupby("HubID")["OrderVolume"].mean()
        non_hol_mean = open_df[open_df["RegionalHoliday"] == 0].groupby("HubID")["OrderVolume"].mean()
        hol_ratio = (hol_mean / non_hol_mean.clip(lower=1.0)).rename("holiday_sales_ratio")

        # Saturday sensitivity ratio
        sat_mean = open_df[open_df["Weekday"] == 6].groupby("HubID")["OrderVolume"].mean()
        weekday_mean = open_df[open_df["Weekday"].isin([1, 2, 3, 4, 5])].groupby("HubID")["OrderVolume"].mean()
        sat_ratio = (sat_mean / weekday_mean.clip(lower=1.0)).rename("saturday_sales_ratio")

        # Operational proportions across all records
        school_pct = train_df.groupby("HubID")["SchoolClosureFlag"].mean().rename("school_hol_pct")
        open_rate = train_df.groupby("HubID")["IsOpen"].mean().rename("open_rate")

        # Assemble ratio matrix
        self.hub_ratios_ = pd.concat(
            [mean_vol, promo_ratio, hol_ratio, sat_ratio, school_pct, open_rate], axis=1
        ).reset_index()

        # Compute global fallbacks for newly observed hubs
        for col in self.hub_ratios_.columns:
            if col != "HubID":
                self.global_defaults_[col] = float(self.hub_ratios_[col].median())

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.hub_ratios_ is not None, "Transformer must be fitted before transform"
        df = df.merge(self.hub_ratios_, on="HubID", how="left")
        
        # Apply global medians for missing ratios
        for col, default_val in self.global_defaults_.items():
            df[col] = df[col].fillna(default_val).astype(np.float32)

        return df
