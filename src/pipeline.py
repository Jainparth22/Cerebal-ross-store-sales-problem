"""
End-to-End Orchestrator Pipeline
================================
CLI entrypoint coordinating data ingestion, memory optimization, feature engineering,
multi-model training, harmonic mean ensembling, and YoY scale calibration.

Usage:
------
python -m src.pipeline --mode train
python -m src.pipeline --mode predict --sample-frac 0.1
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PathConfig, FeatureConfig, ModelParamsConfig, TrainConfig
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.pipeline import FeaturePipeline
from src.models.base import BaseForecaster
from src.models.lgbm_model import LightGBMForecaster
from src.models.xgb_model import XGBoostForecaster
from src.models.catboost_model import CatBoostForecaster
from src.models.summer_specialist import SummerSpecialistForecaster
from src.models.ensemble import HarmonicMeanEnsemble, YoYAnchoredCalibrator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solari Demand Forecasting Engine CLI Orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="predict",
        help="Execution mode: 'train' evaluates CV; 'predict' outputs submission.csv",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional sampling fraction for fast integration testing (0.0 < frac <= 1.0)",
    )
    return parser.parse_args()


def run_pipeline(mode: str = "predict", sample_frac: float = 1.0) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    start_time = time.time()
    print("=" * 80)
    print(">>> STARTING SOLARI DEMAND FORECASTING ENGINE")
    print(f"Mode: {mode.upper()} | Sampling: {sample_frac:.1%}")
    print("=" * 80)

    # 1. Ingestion
    paths = PathConfig()
    loader = DataLoader(paths)
    print("\n[Step 1/6] Ingesting operational datasets...")
    train_df, test_df, meta_df, sample_sub = loader.load_raw_data()
    print(f"  [OK] Raw train: {len(train_df):,} rows | Raw test: {len(test_df):,} rows | Hubs: {len(meta_df):,}")

    if sample_frac < 1.0:
        hubs = meta_df["HubID"].sample(frac=sample_frac, random_state=42).tolist()
        train_df = train_df[train_df["HubID"].isin(hubs)].reset_index(drop=True)
        test_df = test_df[test_df["HubID"].isin(hubs)].reset_index(drop=True)
        meta_df = meta_df[meta_df["HubID"].isin(hubs)].reset_index(drop=True)
        print(f"  * Fast subsample: {len(hubs)} hubs | Train: {len(train_df):,} | Test: {len(test_df):,}")

    # 2. Preprocessing & Calendar Decomposition
    print("\n[Step 2/6] Parsing calendar topologies & structural metadata...")
    preprocessor = DataPreprocessor()
    train_prep, test_prep = preprocessor.transform_pipeline(train_df, test_df, meta_df)

    # Memory downcasting
    train_prep = loader.downcast_dtypes(train_prep, verbose=True)
    test_prep = loader.downcast_dtypes(test_prep, verbose=False)

    # 3. Feature Engineering Engine
    print("\n[Step 3/6] Generating 60+ temporal, elasticity, and lag features...")
    feat_config = FeatureConfig()
    feature_pipeline = FeaturePipeline(feat_config)
    train_fe, test_fe, feature_cols = feature_pipeline.fit_transform_pipeline(train_prep, test_prep)

    # Filter operational training records (IsOpen == 1)
    train_open = train_fe[train_fe["IsOpen"] == 1].reset_index(drop=True)
    X_train = train_open[feature_cols].copy()
    y_train = train_open["OrderVolume"].values

    # Clean non-numeric or infinite columns
    X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
    active_features = list(X_train.columns)
    X_test = test_fe[active_features].select_dtypes(include=[np.number]).fillna(0)
    print(f"  [OK] Final active model predictors: {len(active_features)}")

    # 4. Multi-Model Suite Training
    print("\n[Step 4/6] Training heterogeneous GBDT models on log1p targets...")
    train_config = TrainConfig()

    models = {
        "LightGBM": LightGBMForecaster(),
        "XGBoost": XGBoostForecaster(),
        "CatBoost": CatBoostForecaster(),
        "SummerSpecialist": SummerSpecialistForecaster(),
    }

    test_preds = {}
    for name, model in models.items():
        m_start = time.time()
        print(f"  Training {name}...")
        model.fit(X_train, y_train, early_stopping_rounds=0, verbose=False)
        test_preds[name] = model.predict(X_test)
        print(f"  [OK] {name} fitted in {time.time() - m_start:.1f}s | Test open-mean: {test_preds[name][test_fe['IsOpen']==1].mean():.0f}")

    # 5. Weighted Harmonic Mean Aggregation
    print("\n[Step 5/6] Aggregating predictions via Weighted Harmonic Mean...")
    ensemble_matrix = np.column_stack([test_preds[k] for k in models.keys()])
    harmonic_ensemble = HarmonicMeanEnsemble(train_config.ensemble_weights)
    raw_ensemble = harmonic_ensemble.aggregate(ensemble_matrix)
    print(f"  [OK] Harmonic ensemble open-day raw mean: {raw_ensemble[test_fe['IsOpen']==1].mean():.0f}")

    # Apply multiplicative log-bias calibration estimate
    calibrated_raw = raw_ensemble * train_config.log_bias_multiplier

    # 6. YoY Anchor Shrinkage & Operational Hard Constraint
    print("\n[Step 6/6] Applying YoY Anchor shrinkage & operational invariant filter...")
    calibrator = YoYAnchoredCalibrator(yoy_weight=train_config.yoy_anchor_weight)

    # YoY 364-day lag fallback
    yoy_lags = test_fe["vol_lag_364"].fillna(test_fe["vol_lag_365"]).values
    is_open_flags = test_fe["IsOpen"].values

    final_predictions = calibrator.calibrate(
        ensemble_preds=calibrated_raw,
        yoy_lags=yoy_lags,
        is_open_flags=is_open_flags,
        yoy_growth_rate=1.013,
    )

    open_mask = is_open_flags == 1
    print(f"  [OK] Calibrated open-day mean: {final_predictions[open_mask].mean():.0f}")
    print(f"  [OK] Closed-day zeros enforced: {(final_predictions == 0).sum():,} rows")

    # Save submission
    submission = pd.DataFrame({
        "Id": test_fe["Id"].values if "Id" in test_fe.columns else np.arange(1, len(test_fe) + 1),
        "OrderVolume": np.round(final_predictions, 2),
    })
    submission["OrderVolume"] = submission["OrderVolume"].clip(lower=0.0)
    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Pipeline complete in {time.time() - start_time:.1f}s. Artifact saved to: {out_path.resolve()}")
    print("\nSample Output:")
    print(submission.head())


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(mode=args.mode, sample_frac=args.sample_frac)
