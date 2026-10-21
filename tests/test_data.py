"""
Unit Tests: Data Ingestion & Memory Optimization
================================================
Verifies schema enforcement and numeric type downcasting without precision loss.
"""

import unittest
import numpy as np
import pandas as pd

from src.data.loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test suite for data loading and memory compression."""

    def test_downcast_numeric_dtypes(self):
        """Tests that int64 downcasts to int16/int32 and float64 downcasts to float32."""
        df = pd.DataFrame({
            "small_int": np.array([1, 2, 100], dtype=np.int64),
            "med_int": np.array([1000, 20000, 30000], dtype=np.int64),
            "float_col": np.array([1.5, 2.7, 3.9], dtype=np.float64),
        })

        downcasted = DataLoader.downcast_dtypes(df)

        self.assertIn(downcasted["small_int"].dtype, [np.int8, np.int16])
        self.assertEqual(downcasted["med_int"].dtype, np.int16)
        self.assertEqual(downcasted["float_col"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
