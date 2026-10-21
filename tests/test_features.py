"""
Unit Tests: Feature Engineering & Leakage Prevention
====================================================
Verifies shift-1 causality constraints and annual cyclic lag phase alignment.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.lags import LagRollingTransformer
from src.features.distance import DayCounterTransformer


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature transformations and causality guards."""

    def test_rolling_feature_causality_guard(self):
        """
        Causality Invariant:
        Rolling window statistics for date t must only aggregate history strictly up to t-1.
        A sudden spike in OrderVolume on date t must NOT alter vol_roll7_mean(t).
        """
        dates = pd.date_range("2014-01-01", periods=15, freq="D")
        df = pd.DataFrame({
            "HubID": [1] * 15,
            "Date": dates,
            "OrderVolume": [100.0] * 14 + [999999.0],  # Extreme spike on day 15
        })

        transformer = LagRollingTransformer(lag_days=[7], rolling_windows=[7])
        train_out, _ = transformer.transform_combined(df, pd.DataFrame(columns=["HubID", "Date"]))

        # Day 15 rolling mean must only reflect days 8-14 (all 100.0), unaffected by day 15's spike
        day15_roll = train_out.iloc[14]["vol_roll7_mean"]
        self.assertAlmostEqual(day15_roll, 100.0, places=4)

    def test_yoy_lag_weekday_phase_alignment(self):
        """
        Mathematical Invariant:
        Lag of 364 days (52 * 7) must guarantee exact Day-of-Week equality.
        """
        dates = pd.date_range("2013-01-01", "2015-06-01", freq="D")
        for d in dates:
            d_prev = d - pd.Timedelta(days=364)
            self.assertEqual(d.weekday(), d_prev.weekday(), f"Phase mismatch on {d}")

    def test_day_counter_promo_distance(self):
        """Tests forward and backward promotional distance calculation."""
        dates = pd.date_range("2014-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "HubID": [1] * 5,
            "Date": dates,
            "PromoActive": [0, 0, 1, 0, 0],
            "IsOpen": [1] * 5,
        })

        transformer = DayCounterTransformer()
        train_out, _ = transformer.transform_combined(df, pd.DataFrame(columns=["HubID", "Date"]))

        # On day index 2 (PromoActive=1), DaysSinceLastPromo=0 and DaysUntilNextPromo=0
        self.assertEqual(train_out.iloc[2]["DaysSinceLastPromo"], 0.0)
        self.assertEqual(train_out.iloc[2]["DaysUntilNextPromo"], 0.0)
        # On day index 0 (2 days before promo), DaysUntilNextPromo=2
        self.assertEqual(train_out.iloc[0]["DaysUntilNextPromo"], 2.0)
        # On day index 4 (2 days after promo), DaysSinceLastPromo=2
        self.assertEqual(train_out.iloc[4]["DaysSinceLastPromo"], 2.0)


if __name__ == "__main__":
    unittest.main()
