"""
LightGBM Forecaster Wrapper
===========================
High-capacity gradient boosted decision tree utilizing leaf-wise tree growth
and aggressive column subsampling (feature_fraction = 0.30).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.base import BaseForecaster
from src.config import ModelParamsConfig


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM regressor trained on log1p-transformed targets.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or ModelParamsConfig().lgb_params
        self.model: Optional[lgb.LGBMRegressor] = None
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: int = 150,
        verbose: bool = False,
    ) -> "LightGBMForecaster":
        y_train_log = self.to_log_target(y_train)

        callbacks = []
        if early_stopping_rounds > 0 and eval_set is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
            if verbose:
                callbacks.append(lgb.log_evaluation(period=200))

        eval_data = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_data = [(X_val, self.to_log_target(y_val))]

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train_log,
            eval_set=eval_data,
            callbacks=callbacks if callbacks else None,
        )
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model must be fitted before predict"
        preds_log = self.model.predict(X)
        return self.to_original_scale(preds_log)
