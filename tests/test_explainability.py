import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.forecaster import DemandForecaster
from src.models.explainability import ModelExplainer

@pytest.fixture
def trained_explainer():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(45)]
    records = []
    for d in dates:
        records.append({
            "date": d,
            "product_id": "PROD-EXPLAIN",
            "category": "Apparel",
            "total_quantity": 30 + (d.weekday() == 6) * 20,
            "total_revenue": 1500.0,
            "order_count": 10,
            "avg_price": 50.0,
            "is_promo": bool(d.day % 14 == 0)
        })
    daily_df = pd.DataFrame(records)
    feat_df = FeatureEngineer.generate_demand_features(daily_df)
    X_train, y_train, _, _ = FeatureEngineer.prepare_train_test_split(feat_df, test_days=7)

    fc = DemandForecaster(model_type="xgboost")
    fc.train(X_train, y_train)

    explainer = ModelExplainer(model=fc.model, feature_names=fc.feature_names)
    return explainer, feat_df

def test_global_feature_importance(trained_explainer):
    explainer, feat_df = trained_explainer
    sample = feat_df.head(20)
    importance = explainer.get_global_feature_importance(sample)

    assert len(importance) == len(explainer.feature_names)
    assert all("importance_score" in item for item in importance)
    assert all("friendly_name" in item for item in importance)
    # Check that list is sorted descending
    scores = [item["importance_score"] for item in importance]
    assert scores == sorted(scores, reverse=True)

def test_local_instance_explanation_and_narrative(trained_explainer):
    explainer, feat_df = trained_explainer
    instance = feat_df.iloc[[0]]
    result = explainer.explain_instance(instance)

    assert "prediction" in result
    assert "base_value" in result
    assert "executive_narrative" in result
    assert len(result["executive_narrative"]) > 20
    assert "top_positive_factors" in result
    assert "top_negative_factors" in result
