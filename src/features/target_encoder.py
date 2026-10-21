"""
Out-of-Fold (OOF) Target Encoder
================================
Applies Bayesian smoothed out-of-fold target encoding across hierarchical temporal groups.

Leakage Prevention Invariant:
-----------------------------
Computing naive groupby target means on the full training set introduces optimistic target
leakage into decision tree splits. We apply a 5-fold TimeSeriesSplit walk-forward
loop: each validation partition is encoded strictly from preceding historical folds.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.base import BaseFeatureTransformer


class OutOfFoldTargetEncoder(BaseFeatureTransformer):
    """
    Computes walk-forward out-of-fold target encodings for Hub, Hub x Weekday, and Hub x Month.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 10.0) -> None:
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.full_hub_means_: Dict[int, float] = {}
        self.full_hub_weekday_means_: Dict[Tuple[int, int], float] = {}
        self.full_hub_month_means_: Dict[Tuple[int, int], float] = {}
        self.global_mean_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "OutOfFoldTargetEncoder":
        self.global_mean_ = float(train_df["OrderVolume"].mean())
        
        # Hub overall means
        hub_means = train_df.groupby("HubID")["OrderVolume"].mean().to_dict()
        self.full_hub_means_ = {int(k): float(v) for k, v in hub_means.items()}

        # Hub x Weekday interaction means
        hw_means = train_df.groupby(["HubID", "Weekday"])["OrderVolume"].mean().to_dict()
        self.full_hub_weekday_means_ = {
            (int(k[0]), int(k[1])): float(v) for k, v in hw_means.items()
        }

        # Hub x Month interaction means
        hm_means = train_df.groupby(["HubID", "Month"])["OrderVolume"].mean().to_dict()
        self.full_hub_month_means_ = {
            (int(k[0]), int(k[1])): float(v) for k, v in hm_means.items()
        }

        return self

    def transform_oof_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates out-of-fold encodings on train set using TimeSeriesSplit.
        """
        train_sorted = train_df.sort_values("Date").copy()
        oof_hub = np.full(len(train_sorted), np.nan)
        oof_hw = np.full(len(train_sorted), np.nan)
        oof_hm = np.full(len(train_sorted), np.nan)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for tr_idx, val_idx in tscv.split(train_sorted):
            tr_fold = train_sorted.iloc[tr_idx]

            # Fold encodings with Bayesian smoothing: (n * mean + m * global) / (n + m)
            fold_h = tr_fold.groupby("HubID")["OrderVolume"].agg(["count", "mean"])
            fold_h["smooth"] = (
                fold_h["count"] * fold_h["mean"] + self.smoothing * self.global_mean_
            ) / (fold_h["count"] + self.smoothing)
            h_map = fold_h["smooth"].to_dict()

            fold_hw = tr_fold.groupby(["HubID", "Weekday"])["OrderVolume"].mean().to_dict()
            fold_hm = tr_fold.groupby(["HubID", "Month"])["OrderVolume"].mean().to_dict()

            val_h = train_sorted.iloc[val_idx]["HubID"].map(h_map).values
            val_hw = (
                train_sorted.iloc[val_idx]
                .set_index(["HubID", "Weekday"])
                .index.map(fold_hw)
                .values
            )
            val_hm = (
                train_sorted.iloc[val_idx]
                .set_index(["HubID", "Month"])
                .index.map(fold_hm)
                .values
            )

            oof_hub[val_idx] = val_h
            oof_hw[val_idx] = val_hw
            oof_hm[val_idx] = val_hm

        # Pre-validation warm-up period filled with earliest observed fold mean
        train_sorted["oof_hub_mean"] = pd.Series(oof_hub).bfill().fillna(self.global_mean_).values
        train_sorted["oof_hub_weekday_mean"] = pd.Series(oof_hw).bfill().fillna(self.global_mean_).values
        train_sorted["oof_hub_month_mean"] = pd.Series(oof_hm).bfill().fillna(self.global_mean_).values

        return train_sorted

    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Encodes test set using final fitted full-history target statistics."""
        test_df = test_df.copy()
        test_df["oof_hub_mean"] = test_df["HubID"].map(self.full_hub_means_).fillna(self.global_mean_)
        
        hw_idx = test_df.set_index(["HubID", "Weekday"]).index
        hw_series = pd.Series(hw_idx.map(self.full_hub_weekday_means_), index=test_df.index)
        test_df["oof_hub_weekday_mean"] = hw_series.fillna(test_df["oof_hub_mean"]).astype(np.float32)

        hm_idx = test_df.set_index(["HubID", "Month"]).index
        hm_series = pd.Series(hm_idx.map(self.full_hub_month_means_), index=test_df.index)
        test_df["oof_hub_month_mean"] = hm_series.fillna(test_df["oof_hub_mean"]).astype(np.float32)

        return test_df
