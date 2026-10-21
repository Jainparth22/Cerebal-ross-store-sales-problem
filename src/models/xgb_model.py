"""
XGBoost Forecaster Wrapper
==========================
Histogram-accelerated gradient boosting with depth-wise tree growth
and aggressive column subsampling (colsample_bytree = 0.30).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseForecaster
from src.config import ModelParamsConfig


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost regressor trained on log1p-transformed targets.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or ModelParamsConfig().xgb_params
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: int = 150,
        verbose: bool = False,
    ) -> "XGBoostForecaster":
        y_train_log = self.to_log_target(y_train)

        eval_data = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_data = [(X_val, self.to_log_target(y_val))]

        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
        )
        self.model.fit(
            X_train,
            y_train_log,
            eval_set=eval_data,
            verbose=verbose,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model must be fitted before predict"
        preds_log = self.model.predict(X)
        return self.to_original_scale(preds_log)
