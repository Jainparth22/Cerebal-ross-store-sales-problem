"""
Data Preprocessor Module
========================
Executes deterministic preprocessing, date breakdown, structural missingness resolution,
and metadata joining across train and test partitions.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Orchestrates temporal parsing, feature decomposition, and metadata integration.
    """

    LOYALTY_SCHEDULES: Dict[str, List[int]] = {
        "Jan,Apr,Jul,Oct": [1, 4, 7, 10],
        "Feb,May,Aug,Nov": [2, 5, 8, 11],
        "Mar,Jun,Sept,Dec": [3, 6, 9, 12],
    }

    @staticmethod
    def parse_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes ISO 8601 timestamps into orthogonal calendar features and cyclical projections.
        
        Mathematical / CS Rationale:
        ----------------------------
        Calendar metrics like DayOfWeek (1-7) or Month (1-12) possess cyclical topologies:
        Sunday (7) is adjacent to Monday (1), yet linear or tree representations treat them
        as extreme distant ends of an interval [1, 7].
        Mapping t -> sin(2pi*t/T), cos(2pi*t/T) projects time onto the unit circle S^1,
        preserving continuous distance invariants across periodic boundaries.
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])

        df["Year"] = df["Date"].dt.year.astype(np.int16)
        df["Month"] = df["Date"].dt.month.astype(np.int8)
        df["Day"] = df["Date"].dt.day.astype(np.int8)
        df["DayOfYear"] = df["Date"].dt.dayofyear.astype(np.int16)
        df["Week"] = df["Date"].dt.isocalendar().week.astype(np.int8)
        df["Quarter"] = df["Date"].dt.quarter.astype(np.int8)

        # Operational calendar flags
        df["IsWeekend"] = df["Weekday"].isin([6, 7]).astype(np.int8)
        df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(np.int8)
        df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(np.int8)
        df["IsSummer"] = df["Month"].isin([6, 7, 8]).astype(np.int8)

        # Unit circle projections S^1
        df["sin_weekday"] = np.sin(2 * np.pi * df["Weekday"] / 7).astype(np.float32)
        df["cos_weekday"] = np.cos(2 * np.pi * df["Weekday"] / 7).astype(np.float32)
        df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12).astype(np.float32)
        df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12).astype(np.float32)

        return df

    @classmethod
    def process_metadata(cls, df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges hub metadata and resolves structural vs. stochastic missing values.
        
        Domain Theory:
        --------------
        Missing values in `CompetitorOpenSinceYear` or `LoyaltyProgramSinceWeek`
        are not missing-at-random (MAR); they indicate the structural absence
        of a competitor or non-enrollment in the loyalty program.
        We explicitly encode these structural invariants via indicator flags
        rather than naive mean/mode imputation.
        """
        df = df.merge(meta_df, on="HubID", how="left")

        # Competitor distance features (log1p transform compresses extreme 75km right tail)
        med_dist = df["CompetitorDistance"].median()
        df["CompetitorDistance_clean"] = df["CompetitorDistance"].fillna(med_dist).astype(np.float32)
        df["log_CompetitorDistance"] = np.log1p(df["CompetitorDistance_clean"]).astype(np.float32)
        df["HasCompetitor"] = df["CompetitorOpenSinceYear"].notna().astype(np.int8)

        # Competitor tenure in months
        comp_year = df["CompetitorOpenSinceYear"].fillna(df["Year"])
        comp_month = df["CompetitorOpenSinceMonth"].fillna(df["Month"])
        df["CompetitorAge_months"] = np.maximum(
            0, (df["Year"] - comp_year) * 12 + (df["Month"] - comp_month)
        ).astype(np.float32)

        # Loyalty program tenure in weeks
        df["HasLoyalty"] = (df["LoyaltyProgram"] == 1).astype(np.int8)
        loyalty_year = df["LoyaltyProgramSinceYear"].fillna(df["Year"])
        loyalty_week = df["LoyaltyProgramSinceWeek"].fillna(df["Week"])
        df["LoyaltyAge_weeks"] = np.maximum(
            0, (df["Year"] - loyalty_year) * 52 + (df["Week"] - loyalty_week)
        ).astype(np.float32)

        # Loyalty cycle activity indicator
        def is_loyalty_active_month(row) -> int:
            if row["LoyaltyProgram"] != 1 or pd.isna(row["LoyaltyProgramInterval"]):
                return 0
            schedule = cls.LOYALTY_SCHEDULES.get(str(row["LoyaltyProgramInterval"]).strip(), [])
            return int(row["Month"] in schedule)

        df["LoyaltyActiveMonth"] = df.apply(is_loyalty_active_month, axis=1).astype(np.int8)

        # Holiday composite interaction
        df["AnyHoliday"] = (
            (df["RegionalHoliday"] > 0) | (df["SchoolClosureFlag"] == 1)
        ).astype(np.int8)

        return df

    def transform_pipeline(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs calendar parsing and metadata integration on both train and test partitions."""
        train_processed = self.parse_calendar_features(train_df)
        test_processed = self.parse_calendar_features(test_df)

        train_processed = self.process_metadata(train_processed, meta_df)
        test_processed = self.process_metadata(test_processed, meta_df)

        return train_processed, test_processed
