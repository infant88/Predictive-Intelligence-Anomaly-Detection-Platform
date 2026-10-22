import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.connection import init_db
from src.data_pipeline.generator import DataGenerator
from src.data_pipeline.cleaner import DataCleaner

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    init_db()
    # Ingest a small clean sample to ensure DB has data for endpoints
    raw_df = DataGenerator.generate_raw_ecommerce_data(days=35, include_dirty_data=False)
    cleaned_df, _ = DataCleaner.clean_batch(raw_df)
    DataCleaner.sync_to_database(cleaned_df)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data

def test_api_kpis():
    response = client.get("/api/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "total_revenue" in data
    assert "total_orders" in data
    assert data["total_revenue"] >= 0

def test_api_stream_transaction_event():
    payload = {
        "stock_code": "PROD-TECH-001",
        "description": "Wireless Headphones",
        "quantity": 2.0,
        "unit_price": 149.99,
        "customer_id": "CUST-TEST",
        "country": "US",
        "category": "Electronics"
    }
    response = client.post("/api/data/stream-event", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROCESSED"
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert data["cleaned_transaction"]["total_amount"] == 299.98

def test_api_get_alerts():
    response = client.get("/api/anomalies/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_api_forecast_predict():
    payload = {
        "product_id": "PROD-TECH-001",
        "horizon_days": 7,
        "promo_discount_pct": 10.0
    }
    response = client.post("/api/forecast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "PROD-TECH-001"
    assert len(data["points"]) == 7
    for pt in data["points"]:
        assert pt["lower_bound"] <= pt["predicted_quantity"] <= pt["upper_bound"]
