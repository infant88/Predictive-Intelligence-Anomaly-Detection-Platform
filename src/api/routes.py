from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from src.logger import logger
from src.database.connection import get_db, get_db_session
from src.database.models import AnomalyAlert, CleanedTransaction
from src.database.queries import DatabaseRepository
from src.data_pipeline.generator import DataGenerator
from src.data_pipeline.cleaner import DataCleaner
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.forecaster import DemandForecaster
from src.models.anomaly_detector import AnomalyDetector
from src.models.explainability import ModelExplainer
from src.models.mlflow_tracker import mlflow_tracker
from src.api.schemas import (
    IngestRequest, IngestResponse, QualityAuditResponse,
    StreamTransactionRequest, StreamTransactionResponse,
    ForecastRequest, ForecastResponse, ForecastPoint,
    ExplainRequest, ExplainResponse, FactorImpact,
    AlertItem, AlertUpdateRequest, KPIResponse
)

router = APIRouter()

# Global module state references (cached after training or loading)
forecaster_instance: Optional[DemandForecaster] = None
anomaly_detector_instance: Optional[AnomalyDetector] = None
explainer_instance: Optional[ModelExplainer] = None

def get_or_load_forecaster() -> DemandForecaster:
    global forecaster_instance
    if forecaster_instance is None:
        try:
            forecaster_instance = DemandForecaster.load()
        except Exception:
            logger.warning("No saved forecaster model found. Training fresh model...")
            train_platform_models()
    return forecaster_instance

def get_or_load_detector() -> AnomalyDetector:
    global anomaly_detector_instance
    if anomaly_detector_instance is None:
        try:
            anomaly_detector_instance = AnomalyDetector.load()
        except Exception:
            logger.warning("No saved anomaly detector found. Training fresh detector...")
            train_platform_models()
    return anomaly_detector_instance

def get_or_load_explainer() -> ModelExplainer:
    global explainer_instance
    if explainer_instance is None:
        fc = get_or_load_forecaster()
        explainer_instance = ModelExplainer(model=fc.model, feature_names=fc.feature_names)
    return explainer_instance

def train_platform_models():
    """Trains forecasting, anomaly detection, and explainability models end-to-end."""
    global forecaster_instance, anomaly_detector_instance, explainer_instance

    logger.info("Triggering end-to-end model training...")
    # 1. Fetch cleaned transactions
    from src.database.connection import engine
    from sqlalchemy import text
    try:
        df_clean = pd.read_sql(text("SELECT * FROM cleaned_transactions"), con=engine)
    except Exception:
        df_clean = pd.DataFrame()

    if df_clean.empty:
        # Ingest default data first
        raw_df = DataGenerator.generate_raw_ecommerce_data(days=90, include_dirty_data=True)
        df_clean, _ = DataCleaner.clean_batch(raw_df)
        DataCleaner.sync_to_database(df_clean)

    # 2. Fetch daily metrics
    daily_df = pd.read_sql(text("SELECT * FROM daily_metrics"), con=engine)
    
    # 3. Engineer features
    feat_df = FeatureEngineer.generate_demand_features(daily_df)
    X_train, y_train, X_test, y_test = FeatureEngineer.prepare_train_test_split(feat_df, test_days=14)

    # 4. Train XGBoost Forecaster
    fc = DemandForecaster(model_type="xgboost")
    metrics = fc.train(X_train, y_train, X_test, y_test)
    fc.save()
    forecaster_instance = fc

    # 5. Fit Isolation Forest
    detector = AnomalyDetector(contamination=0.03)
    detector.fit_isolation_forest(df_clean)
    detector.save()
    anomaly_detector_instance = detector

    # Detect initial historical transaction anomalies and persist alerts
    _, txn_alerts = detector.detect_transaction_anomalies(df_clean)
    # Detect time-series sales drops
    ts_alerts: List[Dict[str, Any]] = []
    for prod_id in df_clean["product_id"].unique():
        prod_series = DatabaseRepository.get_product_daily_series(prod_id)
        prod_alerts = detector.detect_sales_drops(prod_series, z_threshold=-2.0)
        ts_alerts.extend(prod_alerts)

    AnomalyDetector.persist_alerts(txn_alerts + ts_alerts)

    # 6. Initialize SHAP Explainer
    explainer_instance = ModelExplainer(model=fc.model, feature_names=fc.feature_names)

    # 7. Log Run to MLflow
    mlflow_tracker.log_training_run(
        model_name="xgboost_demand_forecaster",
        params={"n_estimators": 180, "max_depth": 5, "learning_rate": 0.06},
        metrics=metrics,
        tags={"dataset": "ecommerce_real_world", "features_count": str(len(fc.feature_names))}
    )
    logger.info("End-to-end model training finished successfully.")


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "forecaster": forecaster_instance is not None or "ready",
            "anomaly_detector": anomaly_detector_instance is not None or "ready",
            "mlflow": "active"
        }
    }

@router.get("/kpis", response_model=KPIResponse)
def get_kpis():
    try:
        return DatabaseRepository.get_kpis()
    except Exception as e:
        logger.error(f"KPI calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/ingest-and-clean", response_model=IngestResponse)
def ingest_and_clean_data(payload: IngestRequest):
    """
    Ingests e-commerce data with optional real-world dirty data flaws,
    runs the automated data cleaning and validation pipeline, updates the database,
    and returns a comprehensive data quality audit report.
    """
    try:
        raw_df = DataGenerator.generate_raw_ecommerce_data(
            days=payload.days,
            include_dirty_data=payload.include_dirty_data
        )
        cleaned_df, audit_report = DataCleaner.clean_batch(raw_df)
        DataCleaner.sync_to_database(cleaned_df)

        # Trigger model retraining on newly cleaned data
        train_platform_models()

        return IngestResponse(
            status="SUCCESS",
            message=f"Successfully ingested {len(raw_df)} raw records and cleaned {len(cleaned_df)} transactions.",
            audit_report=QualityAuditResponse(**audit_report.to_dict()),
            total_cleaned_transactions=len(cleaned_df)
        )
    except Exception as e:
        logger.error(f"Data ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {e}")

@router.post("/data/stream-event", response_model=StreamTransactionResponse)
def stream_transaction_event(event: StreamTransactionRequest):
    """
    Streams a single real-time transaction event:
    Cleans and validates the record, scores it with Isolation Forest,
    and triggers an immediate alert if anomalous.
    """
    try:
        raw_dict = event.model_dump()
        if not raw_dict.get("invoice_no"):
            raw_dict["invoice_no"] = f"INV-STRM-{datetime.now().strftime('%H%M%S')}"
        raw_dict["invoice_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cleaned = DataCleaner.clean_single_record(raw_dict)
        if not cleaned:
            raise HTTPException(status_code=400, detail="Transaction rejected by data validation pipeline.")

        detector = get_or_load_detector()
        df_single = pd.DataFrame([cleaned])
        scored_df, alerts = detector.detect_transaction_anomalies(df_single)
        is_anomaly = bool(scored_df["is_anomaly"].iloc[0])
        score = float(scored_df["anomaly_score"].iloc[0])

        alert_dict = None
        if is_anomaly and alerts:
            alert_dict = alerts[0]
            AnomalyDetector.persist_alerts([alert_dict])

        return StreamTransactionResponse(
            status="PROCESSED",
            cleaned_transaction=cleaned,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            alert=alert_dict
        )
    except Exception as e:
        logger.error(f"Stream event error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/predict", response_model=ForecastResponse)
def forecast_demand(request: ForecastRequest):
    """
    Generates multi-day recursive demand forecasts with 95% uncertainty intervals
    and What-If promotion discount simulation.
    """
    try:
        fc = get_or_load_forecaster()
        prod_series = DatabaseRepository.get_product_daily_series(request.product_id)
        if prod_series.empty:
            raise HTTPException(status_code=404, detail=f"No transaction history found for product {request.product_id}")

        forecast_df = fc.forecast_future_product_demand(
            historical_series=prod_series,
            horizon_days=request.horizon_days,
            promo_discount_pct=request.promo_discount_pct,
            persist_db=request.persist_db
        )

        points = [
            ForecastPoint(
                target_date=pd.to_datetime(row.target_date).strftime("%Y-%m-%d"),
                predicted_quantity=float(row.predicted_quantity),
                lower_bound=float(row.lower_bound),
                upper_bound=float(row.upper_bound),
                model_name=str(row.model_name),
                is_promo=bool(row.is_promo),
                confidence=float(row.confidence)
            )
            for row in forecast_df.itertuples()
        ]

        return ForecastResponse(
            product_id=request.product_id,
            horizon_days=request.horizon_days,
            promo_discount_pct=request.promo_discount_pct,
            points=points
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {e}")

@router.post("/forecast/explain", response_model=ExplainResponse)
def explain_prediction(request: ExplainRequest):
    """
    Uses SHAP TreeExplainer to break down the exact factors driving demand predictions
    for a specific product and generates an executive plain-English narrative.
    """
    try:
        explainer = get_or_load_explainer()
        prod_series = DatabaseRepository.get_product_daily_series(request.product_id)
        if prod_series.empty:
            raise HTTPException(status_code=404, detail=f"No history found for product {request.product_id}")

        feat_df = FeatureEngineer.generate_demand_features(prod_series)
        if feat_df.empty:
            raise HTTPException(status_code=400, detail="Insufficient history to construct lag/rolling features.")

        latest_instance = feat_df.iloc[[-1]]
        explanation = explainer.explain_instance(latest_instance)

        return ExplainResponse(
            product_id=request.product_id,
            prediction=explanation["prediction"],
            base_value=explanation["base_value"],
            net_impact=explanation["net_impact"],
            top_positive_factors=[FactorImpact(**f) for f in explanation["top_positive_factors"]],
            top_negative_factors=[FactorImpact(**f) for f in explanation["top_negative_factors"]],
            executive_narrative=explanation["executive_narrative"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explainability error: {e}")

@router.get("/anomalies/alerts", response_model=List[AlertItem])
def get_alerts(limit: int = 50, status: Optional[str] = None):
    """Retrieves anomaly alerts from the database."""
    try:
        raw_alerts = DatabaseRepository.get_recent_alerts(limit=limit, status=status)
        return [AlertItem(**a) for a in raw_alerts]
    except Exception as e:
        logger.error(f"Alert fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/anomalies/alerts/{alert_id}")
def update_alert_status(alert_id: int, payload: AlertUpdateRequest, db: Session = Depends(get_db)):
    """Updates status of an anomaly alert (e.g. ACKNOWLEDGED or RESOLVED)."""
    alert = db.query(AnomalyAlert).filter(AnomalyAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    alert.status = payload.status
    db.commit()
    return {"status": "SUCCESS", "alert_id": alert_id, "new_status": payload.status}

@router.get("/mlflow/runs")
def get_mlflow_runs():
    """Retrieves tracked MLflow experiment runs and model evaluation metrics."""
    try:
        return mlflow_tracker.get_recent_runs()
    except Exception as e:
        logger.error(f"MLflow fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
