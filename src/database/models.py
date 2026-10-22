from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, Index
from src.database.connection import Base

class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_no = Column(String(50), nullable=True)
    stock_code = Column(String(50), nullable=True)
    description = Column(String(255), nullable=True)
    quantity = Column(Float, nullable=True)
    invoice_date = Column(String(50), nullable=True)
    unit_price = Column(Float, nullable=True)
    customer_id = Column(String(50), nullable=True)
    country = Column(String(50), nullable=True)
    category = Column(String(50), nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

class CleanedTransaction(Base):
    __tablename__ = "cleaned_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), nullable=False, unique=True, index=True)
    invoice_no = Column(String(50), nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    customer_id = Column(String(50), nullable=False, index=True)
    country = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    is_weekend = Column(Boolean, default=False)
    is_promo = Column(Boolean, default=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)

class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    total_quantity = Column(Integer, nullable=False)
    total_revenue = Column(Float, nullable=False)
    order_count = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    is_promo = Column(Boolean, default=False)

    __table_args__ = (
        Index("idx_daily_product_date", "product_id", "date"),
    )

class DemandForecast(Base):
    __tablename__ = "demand_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_created_at = Column(DateTime, default=datetime.utcnow, index=True)
    target_date = Column(DateTime, nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    category = Column(String(50), nullable=False)
    predicted_quantity = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    model_name = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.95)

class AnomalyAlert(Base):
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False, index=True)  # 'SALES_DROP', 'SUSPICIOUS_TRANSACTION', 'ABNORMAL_ACTIVITY'
    entity_id = Column(String(100), nullable=False)              # product_id or invoice_no or customer_id
    metric_name = Column(String(50), nullable=False)
    actual_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    deviation_pct = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, index=True)    # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    description = Column(Text, nullable=False)
    action_item = Column(Text, nullable=False)
    status = Column(String(20), default="OPEN")                  # 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'
