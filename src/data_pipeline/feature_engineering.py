from typing import List, Tuple
import numpy as np
import pandas as pd
from src.logger import logger

class FeatureEngineer:
    """Constructs time-series lag, rolling window, calendar, and pricing features for ML modeling."""

    FEATURE_COLS = [
        "lag_1", "lag_2", "lag_7", "lag_14", "lag_21",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_max_7", "rolling_min_7",
        "day_of_week", "day_of_month", "month", "is_weekend",
        "avg_price", "is_promo", "price_ratio_7d"
    ]

    @classmethod
    def generate_demand_features(
        cls,
        daily_df: pd.DataFrame,
        target_col: str = "total_quantity"
    ) -> pd.DataFrame:
        """
        Creates time-series lag and rolling statistics per product.
        Input dataframe must contain: ['date', 'product_id', 'total_quantity', 'avg_price', 'is_promo'].
        """
        logger.info(f"Engineering features for {len(daily_df)} daily records...")
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)

        feature_dfs: List[pd.DataFrame] = []

        # Process per product group to avoid data leakage between products
        for product_id, group in df.groupby("product_id"):
            grp = group.copy().set_index("date")

            # Ensure continuous daily index
            idx = pd.date_range(start=grp.index.min(), end=grp.index.max(), freq="D")
            grp = grp.reindex(idx)
            grp["product_id"] = product_id
            if "category" in grp.columns:
                grp["category"] = grp["category"].bfill().ffill()

            grp[target_col] = grp[target_col].fillna(0)
            grp["avg_price"] = grp["avg_price"].bfill().ffill()
            grp["is_promo"] = grp["is_promo"].fillna(False).astype(int)

            # Lags
            for lag in [1, 2, 7, 14, 21]:
                grp[f"lag_{lag}"] = grp[target_col].shift(lag)

            # Rolling stats (closed='left' to prevent lookahead leakage)
            rolling_7 = grp[target_col].shift(1).rolling(window=7, min_periods=3)
            grp["rolling_mean_7"] = rolling_7.mean()
            grp["rolling_std_7"] = rolling_7.std().fillna(0)
            grp["rolling_max_7"] = rolling_7.max()
            grp["rolling_min_7"] = rolling_7.min()

            rolling_14 = grp[target_col].shift(1).rolling(window=14, min_periods=5)
            grp["rolling_mean_14"] = rolling_14.mean()

            # Calendar features
            grp["day_of_week"] = grp.index.weekday
            grp["day_of_month"] = grp.index.day
            grp["month"] = grp.index.month
            grp["is_weekend"] = grp["day_of_week"].isin([5, 6]).astype(int)

            # Pricing ratio relative to past 7-day average
            rolling_price_7 = grp["avg_price"].shift(1).rolling(window=7, min_periods=1).mean()
            grp["price_ratio_7d"] = (grp["avg_price"] / (rolling_price_7 + 1e-6)).fillna(1.0)

            grp = grp.reset_index().rename(columns={"index": "date"})
            feature_dfs.append(grp)

        combined_df = pd.concat(feature_dfs, ignore_index=True)
        # Drop rows where lag_21 is NaN because of initial window warmup
        cleaned_feature_df = combined_df.dropna(subset=["lag_21", "rolling_mean_14"]).reset_index(drop=True)
        logger.info(f"Generated {len(cleaned_feature_df)} feature vectors ready for training.")
        return cleaned_feature_df

    @classmethod
    def prepare_train_test_split(
        cls,
        feature_df: pd.DataFrame,
        test_days: int = 14,
        target_col: str = "total_quantity"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Performs chronological train/test split to prevent temporal leakage."""
        cutoff_date = feature_df["date"].max() - pd.Timedelta(days=test_days)
        train_mask = feature_df["date"] <= cutoff_date
        test_mask = feature_df["date"] > cutoff_date

        X_train = feature_df.loc[train_mask, cls.FEATURE_COLS]
        y_train = feature_df.loc[train_mask, target_col]
        X_test = feature_df.loc[test_mask, cls.FEATURE_COLS]
        y_test = feature_df.loc[test_mask, target_col]

        logger.info(f"Split data into Train: {len(X_train)} samples, Test: {len(X_test)} samples.")
        return X_train, y_train, X_test, y_test
