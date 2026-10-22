import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.forecaster import DemandForecaster

@pytest.fixture
def trained_forecaster():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(50)]
    records = []
    for d in dates:
        records.append({
            "date": d,
            "product_id": "PROD-TECH-001",
            "category": "Electronics",
            "total_quantity": 40 + (d.weekday() == 5 or d.weekday() == 6) * 15 + np.random.randint(-2, 3),
            "total_revenue": 2000.0,
            "order_count": 15,
            "avg_price": 100.0,
            "is_promo": bool(d.day % 10 == 0)
        })
    daily_df = pd.DataFrame(records)
    feat_df = FeatureEngineer.generate_demand_features(daily_df)
    X_train, y_train, X_test, y_test = FeatureEngineer.prepare_train_test_split(feat_df, test_days=10)

    fc = DemandForecaster(model_type="xgboost")
    metrics = fc.train(X_train, y_train, X_test, y_test)
    return fc, daily_df, metrics

def test_forecaster_trains_and_evaluates(trained_forecaster):
    fc, _, metrics = trained_forecaster
    assert fc.model is not None
    assert "test_rmse" in metrics
    assert "test_mae" in metrics
    assert "test_r2" in metrics
    assert metrics["test_rmse"] > 0

def test_recursive_multi_step_forecasting(trained_forecaster):
    fc, daily_df, _ = trained_forecaster
    hist_series = daily_df.rename(columns={"total_quantity": "quantity"})
    horizon = 14
    forecast_df = fc.forecast_future_product_demand(
        historical_series=hist_series,
        horizon_days=horizon,
        promo_discount_pct=0.0
    )

    assert len(forecast_df) == horizon
    assert "predicted_quantity" in forecast_df.columns
    assert "lower_bound" in forecast_df.columns
    assert "upper_bound" in forecast_df.columns
    # Check that bounds are logical
    for row in forecast_df.itertuples():
        assert row.lower_bound <= row.predicted_quantity <= row.upper_bound

def test_what_if_promotion_uplift(trained_forecaster):
    fc, daily_df, _ = trained_forecaster
    hist_series = daily_df.rename(columns={"total_quantity": "quantity"})
    baseline = fc.forecast_future_product_demand(hist_series, horizon_days=7, promo_discount_pct=0.0)
    promo = fc.forecast_future_product_demand(hist_series, horizon_days=7, promo_discount_pct=25.0)

    # Promo demand should be higher than baseline demand due to price elasticity
    assert promo["predicted_quantity"].sum() > baseline["predicted_quantity"].sum()
