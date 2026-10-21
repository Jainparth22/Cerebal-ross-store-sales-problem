"""
Base Feature Transformer Module
===============================
Defines the abstract interface for feature transformers following the Scikit-Learn
Transformer design pattern (fit / transform protocol).
"""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Abstract Base Class for feature transformers.
    Enforces clear separation between state estimation (fit) and projection (transform).
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseFeatureTransformer":
        """
        Calculates and stores feature parameters strictly from training history.
        Must NOT see test data to prevent target or distribution leakage.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned transformations to an arbitrary partition (train or test).
        """
        raise NotImplementedError

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method fitting on train and returning transformed train."""
        return self.fit(train_df).transform(train_df)
