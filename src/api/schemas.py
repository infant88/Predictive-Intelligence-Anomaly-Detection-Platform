from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    days: int = Field(default=90, ge=30, le=365, description="Number of historical days of e-commerce data to ingest")
    include_dirty_data: bool = Field(default=True, description="Whether to inject real-world dirty data flaws for cleaning validation")

class QualityAuditResponse(BaseModel):
    total_raw_records: int
    duplicates_removed: int
    cancellations_handled: int
    missing_customers_imputed: int
    missing_descriptions_imputed: int
    invalid_prices_removed: int
    invalid_quantities_removed: int
    final_cleaned_records: int
    clean_retention_rate_pct: float
    validation_checks: Dict[str, bool]

class IngestResponse(BaseModel):
    status: str
    message: str
    audit_report: QualityAuditResponse
    total_cleaned_transactions: int

class StreamTransactionRequest(BaseModel):
    invoice_no: Optional[str] = None
    stock_code: str = Field(..., json_schema_extra={"example": "PROD-TECH-001"})
    description: Optional[str] = "Wireless Noise-Cancelling Headphones"
    quantity: float = Field(..., json_schema_extra={"example": 2.0})
    unit_price: float = Field(..., json_schema_extra={"example": 149.99})
    customer_id: Optional[str] = "CUST-1045"
    country: Optional[str] = "United States"
    category: Optional[str] = "Electronics"

class StreamTransactionResponse(BaseModel):
    status: str
    cleaned_transaction: Dict[str, Any]
    is_anomaly: bool
    anomaly_score: float
    alert: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "PROD-TECH-001"})
    horizon_days: int = Field(default=14, ge=1, le=60)
    promo_discount_pct: float = Field(default=0.0, ge=0.0, le=90.0, description="What-if simulated promotional discount")
    persist_db: bool = False

class ForecastPoint(BaseModel):
    target_date: str
    predicted_quantity: float
    lower_bound: float
    upper_bound: float
    model_name: str
    is_promo: bool
    confidence: float

class ForecastResponse(BaseModel):
    product_id: str
    horizon_days: int
    promo_discount_pct: float
    points: List[ForecastPoint]

class FactorImpact(BaseModel):
    feature: str
    friendly_name: str
    actual_feature_value: float
    shap_impact: float
    direction: str

class ExplainRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "PROD-TECH-001"})

class ExplainResponse(BaseModel):
    product_id: str
    prediction: float
    base_value: float
    net_impact: float
    top_positive_factors: List[FactorImpact]
    top_negative_factors: List[FactorImpact]
    executive_narrative: str

class AlertItem(BaseModel):
    id: int
    detected_at: str
    alert_type: str
    entity_id: str
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_pct: float
    anomaly_score: float
    severity: str
    description: str
    action_item: str
    status: str

class AlertUpdateRequest(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ACKNOWLEDGED"})

class KPIResponse(BaseModel):
    total_orders: int
    total_revenue: float
    total_units_sold: int
    total_customers: int
    total_products: int
    open_alerts: int
    critical_alerts: int
    high_alerts: int
