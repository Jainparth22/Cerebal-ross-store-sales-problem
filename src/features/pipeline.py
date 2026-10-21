"""
Feature Pipeline Orchestrator (Composite Pattern)
=================================================
Composes individual domain transformers into an atomic, deterministic feature generation pipeline.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np

from src.config import FeatureConfig
from src.features.ratios import HubRatioTransformer
from src.features.trend import HubTrendTransformer
from src.features.distance import DayCounterTransformer
from src.features.lags import LagRollingTransformer
from src.features.target_encoder import OutOfFoldTargetEncoder


class FeaturePipeline:
    """
    Composite transformer coordinating the complete 60+ feature extraction workflow.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()) -> None:
        self.config = config
        self.ratio_transformer = HubRatioTransformer()
        self.trend_transformer = HubTrendTransformer(
            lookback_days=config.trend_lookback_days, alpha=config.ridge_alpha
        )
        self.distance_transformer = DayCounterTransformer()
        self.lag_transformer = LagRollingTransformer(
            lag_days=config.lag_days, rolling_windows=config.rolling_windows
        )
        self.target_encoder = OutOfFoldTargetEncoder()

    def fit_transform_pipeline(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Executes end-to-end feature extraction across train and test partitions.
        
        Returns:
            Tuple of (train_fe, test_fe, feature_columns)
        """
        print("  [1/5] Extracting hub elasticity & ratio aggregates...")
        self.ratio_transformer.fit(train_df)
        train_df = self.ratio_transformer.transform(train_df)
        test_df = self.ratio_transformer.transform(test_df)

        print("  [2/5] Fitting per-hub Ridge regression sales trend...")
        self.trend_transformer.fit(train_df)
        train_df = self.trend_transformer.transform(train_df)
        test_df = self.trend_transformer.transform(test_df)

        print("  [3/5] Calculating promotional distance & reopening surge counters...")
        train_df, test_df = self.distance_transformer.transform_combined(train_df, test_df)

        print("  [4/5] Computing annual cyclic lags & causal rolling statistics...")
        train_df, test_df = self.lag_transformer.transform_combined(train_df, test_df)

        print("  [5/5] Performing TimeSeriesSplit out-of-fold target encoding...")
        self.target_encoder.fit(train_df)
        train_df = self.target_encoder.transform_oof_train(train_df)
        test_df = self.target_encoder.transform(test_df)

        # Assemble eligible feature columns
        feature_cols = [
            c for c in train_df.columns
            if c not in self.config.drop_columns
            and not pd.api.types.is_datetime64_any_dtype(train_df[c])
        ]

        print(f"  [OK] Feature pipeline complete. Total model predictors: {len(feature_cols)}")
        return train_df, test_df, feature_cols
