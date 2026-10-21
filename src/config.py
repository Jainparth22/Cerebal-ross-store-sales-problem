"""
Configuration Management Module
===============================
Provides immutable, type-annotated configuration dataclasses for file paths,
feature engineering hyperparameters, tree ensemble parameters, and cross-validation splits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass(frozen=True)
class PathConfig:
    """System paths for data ingestion, model artifacts, and test submissions."""
    data_dir: Path = Path("Dataset")
    train_file: str = "orders_train.csv"
    test_file: str = "orders_test.csv"
    meta_file: str = "hub_metadata.csv"
    sample_sub_file: str = "sample_submission.csv"
    output_dir: Path = Path("artifacts")
    submission_path: Path = Path("submission.csv")

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def meta_path(self) -> Path:
        return self.data_dir / self.meta_file

    @property
    def sample_sub_path(self) -> Path:
        return self.data_dir / self.sample_sub_file


@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparameters and column specifications for feature generation."""
    # Annual cyclic lags (days)
    lag_days: List[int] = field(default_factory=lambda: [7, 14, 21, 28, 364, 365, 371, 728])
    # Rolling statistic windows (days)
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    # Ridge regression trend lookback window (days)
    trend_lookback_days: int = 365
    ridge_alpha: float = 1.0
    
    # Categorical columns
    cat_columns: List[str] = field(default_factory=lambda: [
        "HubFormat", "AssortmentTier", "LoyaltyProgram", "IsOpen",
        "PromoActive", "RegionalHoliday", "SchoolClosureFlag", "Weekday",
        "IsWeekend", "AnyHoliday", "LoyaltyActiveMonth", "OpenAfterClosed",
        "Quarter", "IsSummer"
    ])

    # Columns excluded from model training matrix
    drop_columns: List[str] = field(default_factory=lambda: [
        "Date", "OrderVolume", "AppSessions",
        "CompetitorOpenSinceMonth", "CompetitorOpenSinceYear",
        "LoyaltyProgramSinceWeek", "LoyaltyProgramSinceYear",
        "LoyaltyProgramInterval", "CompetitorDistance", "Id", "closed_streak"
    ])


@dataclass(frozen=True)
class ModelParamsConfig:
    """Hyperparameters for Gradient Boosted Decision Tree (GBDT) algorithms."""
    # LightGBM configuration
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "min_child_samples": 40,
        "feature_fraction": 0.30,  # Aggressive column subsampling to decorrelate trees
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    })

    # XGBoost configuration
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "learning_rate": 0.02,
        "max_depth": 7,
        "n_estimators": 5000,
        "subsample": 0.85,
        "colsample_bytree": 0.30,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    })

    # CatBoost configuration
    cb_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 5000,
        "learning_rate": 0.02,
        "depth": 7,
        "rsm": 0.30,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1
    })


@dataclass(frozen=True)
class TrainConfig:
    """Training, cross-validation, and ensembling strategy configuration."""
    n_splits: int = 5
    early_stopping_rounds: int = 150
    # Weights for Harmonic Mean Ensemble: [LightGBM, XGBoost, CatBoost, SummerSpecialist]
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4275, 0.3147, 0.0950, 0.1628])
    # Multiplicative log-bias calibration estimate
    log_bias_multiplier: float = 1.008
    # YoY Anchor blend weight: alpha * YoY_pred + (1 - alpha) * Ensemble_pred
    yoy_anchor_weight: float = 0.80
    random_seed: int = 42
