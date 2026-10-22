import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.anomaly_detector import AnomalyDetector

def test_detect_sales_drop_identifies_outage():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    records = []
    for i, d in enumerate(dates):
        # On days 24 and 25, sales drop from 100 to 5 (e.g. server crash)
        qty = 5 if i in (24, 25) else 100 + np.random.randint(-5, 5)
        records.append({
            "date": d,
            "product_id": "PROD-TEST",
            "quantity": qty,
            "total_revenue": qty * 20.0
        })
    df = pd.DataFrame(records)

    detector = AnomalyDetector()
    alerts = detector.detect_sales_drops(df, z_threshold=-2.0, rolling_window=14)

    assert len(alerts) >= 1
    drop_alert = alerts[0]
    assert drop_alert["alert_type"] == "SALES_DROP"
    assert drop_alert["severity"] in ("CRITICAL", "HIGH")
    assert drop_alert["actual_value"] == 5

def test_isolation_forest_transaction_anomaly_scoring():
    # 98 normal transactions and 2 extreme bot attacks
    records = []
    for i in range(100):
        if i in (10, 50):
            # Outlier: huge quantity and price
            records.append({"quantity": 800, "unit_price": 500.0, "total_amount": 400000.0, "invoice_no": f"INV-FRAUD-{i}", "customer_id": "BOT"})
        else:
            qty = np.random.randint(1, 5)
            price = 49.99
            records.append({"quantity": qty, "unit_price": price, "total_amount": qty * price, "invoice_no": f"INV-{i}", "customer_id": f"C-{i}"})

    df = pd.DataFrame(records)
    detector = AnomalyDetector(contamination=0.03)
    detector.fit_isolation_forest(df)

    scored_df, alerts = detector.detect_transaction_anomalies(df)
    assert "anomaly_score" in scored_df.columns
    assert "is_anomaly" in scored_df.columns
    assert len(alerts) >= 1
    # Fraud transactions should have high anomaly scores
    fraud_scores = scored_df.loc[scored_df["customer_id"] == "BOT", "anomaly_score"]
    assert all(s > 0.6 for s in fraud_scores)
