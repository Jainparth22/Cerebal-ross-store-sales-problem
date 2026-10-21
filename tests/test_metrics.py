"""
Unit Tests: Mathematical Invariants & Evaluation Metrics
========================================================
Proves metric compliance and verifies that Harmonic Mean <= Arithmetic Mean (Jensen's inequality).
"""

import unittest
import numpy as np

from src.models.base import BaseForecaster
from src.models.ensemble import HarmonicMeanEnsemble, YoYAnchoredCalibrator


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for mathematical and statistical properties."""

    def test_rmsle_zero_on_exact_match(self):
        """RMSLE must evaluate to exactly 0.0 when predictions equal targets."""
        y_true = np.array([100.0, 500.0, 1000.0, 0.0])
        y_pred = np.array([100.0, 500.0, 1000.0, 0.0])
        score = BaseForecaster.rmsle(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_rmsle_clips_negative_predictions(self):
        """Predictions below zero must be clamped safely without NaN/exception."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([-5.0, 20.0])
        score = BaseForecaster.rmsle(y_true, y_pred)
        self.assertFalse(np.isnan(score))
        self.assertGreater(score, 0.0)

    def test_jensen_inequality_harmonic_le_arithmetic(self):
        """
        Theorem (Harmonic-Arithmetic Mean Inequality):
        For any positive sequence x_i > 0 and convex weights w_i >= 0 (sum w_i = 1):
            H(x, w) <= A(x, w)
        with equality iff x_1 = x_2 = ... = x_M.
        """
        np.random.seed(42)
        # Generate 1000 synthetic multi-model prediction vectors
        preds_matrix = np.random.uniform(500.0, 15000.0, size=(1000, 4))
        weights = [0.4, 0.3, 0.1, 0.2]

        ensemble = HarmonicMeanEnsemble(weights)
        h_mean = ensemble.aggregate(preds_matrix)
        a_mean = np.dot(preds_matrix, np.array(weights) / sum(weights))

        # Check H <= A for all 1000 test vectors
        self.assertTrue(np.all(h_mean <= a_mean + 1e-7))
        # Ensure strict inequality when models disagree
        self.assertTrue(np.any(h_mean < a_mean))

    def test_yoy_calibrator_enforces_closed_hub_zero(self):
        """Operational invariant: IsOpen == 0 must strictly yield 0.0."""
        calibrator = YoYAnchoredCalibrator(yoy_weight=0.80)
        ensemble_preds = np.array([5000.0, 7500.0, 9000.0])
        yoy_lags = np.array([4800.0, 7200.0, 8500.0])
        is_open = np.array([1, 0, 1])

        calibrated = calibrator.calibrate(ensemble_preds, yoy_lags, is_open)
        self.assertEqual(calibrated[1], 0.0)
        self.assertGreater(calibrated[0], 0.0)
        self.assertGreater(calibrated[2], 0.0)


if __name__ == "__main__":
    unittest.main()
