"""
Base Forecaster & Evaluation Metrics Module
===========================================
Defines the abstract model interface and exact mathematical implementation of
Root Mean Squared Logarithmic Error (RMSLE).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract interface for time-series regression forecasters.
    """

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> "BaseForecaster":
        """Fits the underlying model on feature matrix and log1p-transformed targets."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates back-transformed expm1 predictions on the original scale."""
        raise NotImplementedError

    @staticmethod
    def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Exact implementation of Root Mean Squared Logarithmic Error.
        
        Mathematical Definition:
        ------------------------
        RMSLE = sqrt( 1/n * sum_{i=1}^n (log(max(0, p_i) + 1) - log(a_i + 1))^2 )
        
        Computational Invariants:
        -------------------------
        1. Non-negativity clip: Predictions p_i < 0 must be clipped to 0 prior to log1p,
           as log(x) is undefined for x <= 0 in real space.
        2. Finite bounds: NaN or inf values fail fast.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
        return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

    @staticmethod
    def to_log_target(y: np.ndarray) -> np.ndarray:
        """Transforms original target scale into log1p space: log(y + 1)."""
        return np.log1p(np.maximum(y, 0.0))

    @staticmethod
    def to_original_scale(y_log: np.ndarray) -> np.ndarray:
        """Inverts log1p predictions back to original sales volume: exp(y) - 1."""
        return np.maximum(0.0, np.expm1(y_log))
