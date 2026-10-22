import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_pipeline.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_daily_data():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(40)]
    records = []
    for d in dates:
        records.append({
            "date": d,
            "product_id": "TEST-PROD",
            "category": "Electronics",
            "total_quantity": 50 + int(10 * np.sin(d.day)),
            "total_revenue": 500.0,
            "order_count": 10,
            "avg_price": 50.0,
            "is_promo": bool(d.day % 7 == 0)
        })
    return pd.DataFrame(records)

def test_feature_generation_creates_expected_columns(sample_daily_data):
    feat_df = FeatureEngineer.generate_demand_features(sample_daily_data)
    assert not feat_df.empty
    for col in FeatureEngineer.FEATURE_COLS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
    assert feat_df[FeatureEngineer.FEATURE_COLS].isna().sum().sum() == 0

def test_train_test_split_temporal_integrity(sample_daily_data):
    feat_df = FeatureEngineer.generate_demand_features(sample_daily_data)
    test_days = 7
    X_train, y_train, X_test, y_test = FeatureEngineer.prepare_train_test_split(feat_df, test_days=test_days)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(feat_df)
    assert len(X_train.columns) == len(FeatureEngineer.FEATURE_COLS)
