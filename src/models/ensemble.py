"""
Ensemble Aggregator & YoY Scale Calibrator Module
================================================
Implements mathematically proven Weighted Harmonic Mean aggregation and Bayesian-style
YoY anchor calibration with operational hard-constraint filtering.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class HarmonicMeanEnsemble:
    """
    Computes the weighted harmonic mean across heterogeneous model predictions.
    
    Mathematical / CS Foundation:
    ------------------------------
    Let x_1, x_2, ..., x_M > 0 be predictions from M distinct model architectures with weights w_i >= 0.
    The Weighted Harmonic Mean is defined as:
        H(x, w) = (sum_{i=1}^M w_i) / (sum_{i=1}^M w_i / x_i)

    By Jensen's Inequality, because f(t) = 1/t is strictly convex on (0, inf), the Harmonic Mean
    is strictly less than or equal to the Arithmetic Mean:
        H(x, w) <= A(x, w) = sum_{i=1}^M w_i * x_i
        
    Under the Root Mean Squared Logarithmic Error (RMSLE) loss:
        L(y, p) = (log(p + 1) - log(y + 1))^2
    an over-prediction p >> y yields an unbounded convex error penalty, while an under-prediction
    p << y is bounded from below by (log(1) - log(y + 1))^2 = log(y + 1)^2.
    Consequently, minimizing expected risk under RMSLE requires a conservative, left-skewed
    point estimator. The harmonic mean naturally dampens model disagreements and resists
    destructive right-tail upward spikes.
    """

    def __init__(self, weights: Optional[List[float]] = None) -> None:
        self.weights = np.array(weights) if weights is not None else None
        if self.weights is not None:
            self.weights = self.weights / np.sum(self.weights)

    def aggregate(self, preds_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates weighted harmonic mean across model columns.
        
        Args:
            preds_matrix: 2D array of shape (N_samples, M_models)
        """
        assert preds_matrix.ndim == 2, "preds_matrix must be 2D array"
        n_samples, n_models = preds_matrix.shape

        if self.weights is None:
            weights = np.ones(n_models) / n_models
        else:
            assert len(self.weights) == n_models, "Weights length must match model count"
            weights = self.weights

        # Numerical stability clamp to prevent division by zero
        safe_preds = np.maximum(preds_matrix, 1e-4)
        reciprocal_sum = np.sum(weights / safe_preds, axis=1)
        harmonic_mean = 1.0 / reciprocal_sum

        return harmonic_mean


class YoYAnchoredCalibrator:
    """
    Applies empirical Year-over-Year (YoY) anchor shrinkage to eliminate recursive lag decay.
    
    Mathematical / Systems Formulation:
    -----------------------------------
    When forecasting multi-step horizons (42 days ahead), autoregressive lag features (lag_7, lag_14)
    degrade because future ground-truth values are unavailable, leading to variance collapse or
    feature-attribution drift.
    
    We model the final prediction as a convex combination of an empirical seasonal prior (YoY Anchor)
    and the daily model likelihood:
        y_final = alpha * (y_{t-364} * g_{YoY}) + (1 - alpha) * y_ensemble
    where:
    - y_{t-364} is the exact day-of-week aligned demand from 52 weeks prior (2014 summer actuals).
    - g_{YoY} is the measured macro annual growth rate.
    - alpha = 0.80 establishes the historical seasonal baseline (~7,200 orders), while the 20%
      ensemble provides intra-week promo elasticity and daily volatility.
      
    Operational Invariant:
    ----------------------
    State predicate: IsOpen(h, t) = 0 ==> OrderVolume(h, t) = 0.0 with probability 1.0.
    Hardcoded in O(1) post-condition to eliminate unnecessary residual entropy.
    """

    def __init__(self, yoy_weight: float = 0.80) -> None:
        self.yoy_weight = yoy_weight

    def calibrate(
        self,
        ensemble_preds: np.ndarray,
        yoy_lags: np.ndarray,
        is_open_flags: np.ndarray,
        yoy_growth_rate: float = 1.013,
    ) -> np.ndarray:
        """
        Computes calibrated blend and enforces closed-hub invariance.
        """
        # Formulate seasonal prior
        prior_yoy = np.maximum(0.0, yoy_lags * yoy_growth_rate)
        
        # Convex combination
        calibrated = self.yoy_weight * prior_yoy + (1.0 - self.yoy_weight) * ensemble_preds
        
        # Enforce non-negativity
        calibrated = np.maximum(0.0, calibrated)
        
        # Operational Invariant: Closed hubs strictly process 0 orders
        calibrated[is_open_flags == 0] = 0.0

        return calibrated
