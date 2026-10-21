"""
CatBoost Forecaster Wrapper
===========================
Symmetric oblivious decision tree boosting with random subspace method (rsm = 0.30).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.base import BaseForecaster
from src.config import ModelParamsConfig


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost regressor trained on log1p-transformed targets.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        cat_features: Optional[List[str]] = None,
    ) -> None:
        self.params = params or ModelParamsConfig().cb_params
        self.cat_features = cat_features or []
        self.model: Optional[CatBoostRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: int = 150,
        verbose: bool = False,
    ) -> "CatBoostForecaster":
        y_train_log = self.to_log_target(y_train)

        eval_data = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_data = (X_val, self.to_log_target(y_val))

        # Filter valid cat features present in X_train
        cats = [c for c in self.cat_features if c in X_train.columns]

        self.model = CatBoostRegressor(
            **self.params,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
            cat_features=cats if cats else None,
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
