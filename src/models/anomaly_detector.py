import os
import joblib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR
from src.logger import logger
from src.database.connection import get_db_session
from src.database.models import AnomalyAlert

class AnomalyDetector:
    """
    Dual-layer anomaly detection engine:
    1. Time-series statistical anomaly detection (Sudden sales drops, demand crashes, surges).
    2. Machine Learning Isolation Forest (Suspicious transactions, fraud bursts, price abuse).
    """

    TRANSACTION_FEATURES = ["quantity", "unit_price", "total_amount"]

    def __init__(self, contamination: float = 0.03):
        self.contamination = contamination
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ==========================================
    # Layer 1: Time-Series Sales Drop Detection
    # ==========================================
    @staticmethod
    def detect_sales_drops(
        daily_series: pd.DataFrame,
        z_threshold: float = -2.0,
        rolling_window: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Scans time-series sales for sudden drops or abnormal slumps using rolling stats.
        Flags critical revenue drops, stockouts, and outage events.
        """
        df = daily_series.sort_values(by="date").copy()
        if len(df) < rolling_window:
            return []

        df["rolling_mean"] = df["quantity"].rolling(window=rolling_window, min_periods=5).mean()
        df["rolling_std"] = df["quantity"].rolling(window=rolling_window, min_periods=5).std().replace(0, 1e-6)
        df["z_score"] = (df["quantity"] - df["rolling_mean"]) / df["rolling_std"]
        df["pct_change_vs_expected"] = (
            (df["quantity"] - df["rolling_mean"]) / df["rolling_mean"].replace(0, 1e-6)
        ) * 100

        detected_alerts: List[Dict[str, Any]] = []

        for row in df.itertuples():
            if pd.isna(row.z_score) or pd.isna(row.rolling_mean):
                continue

            # Check for sudden sales drops
            if row.z_score <= z_threshold:
                pct_drop = abs(round(row.pct_change_vs_expected, 1))
                if row.z_score <= -3.0 or pct_drop >= 75:
                    severity = "CRITICAL"
                    action = "Immediate action: Verify checkout/payment gateway logs and check physical warehouse stock availability."
                elif row.z_score <= -2.5 or pct_drop >= 50:
                    severity = "HIGH"
                    action = "Investigate product page traffic, competitor price undercut, or recent UI deploy errors."
                else:
                    severity = "MEDIUM"
                    action = "Monitor demand for next 24 hours. Check ad spend and search ranking stability."

                detected_alerts.append({
                    "detected_at": row.date if isinstance(row.date, datetime) else pd.to_datetime(row.date),
                    "alert_type": "SALES_DROP",
                    "entity_id": getattr(row, "product_id", "ALL_PRODUCTS"),
                    "metric_name": "daily_sales_quantity",
                    "actual_value": float(row.quantity),
                    "expected_value": float(round(row.rolling_mean, 1)),
                    "deviation_pct": -pct_drop,
                    "anomaly_score": round(abs(float(row.z_score)), 3),
                    "severity": severity,
                    "description": f"Sudden sales drop of {pct_drop}% detected. Actual: {row.quantity} units (Expected ~{round(row.rolling_mean, 1)} units, Z={round(row.z_score, 2)}).",
                    "action_item": action,
                    "status": "OPEN"
                })

            # Check for sudden massive spikes (abnormal burst or scalper hoarding)
            elif row.z_score >= 3.5:
                spike_pct = round(row.pct_change_vs_expected, 1)
                detected_alerts.append({
                    "detected_at": row.date if isinstance(row.date, datetime) else pd.to_datetime(row.date),
                    "alert_type": "ABNORMAL_ACTIVITY",
                    "entity_id": getattr(row, "product_id", "ALL_PRODUCTS"),
                    "metric_name": "daily_sales_spike",
                    "actual_value": float(row.quantity),
                    "expected_value": float(round(row.rolling_mean, 1)),
                    "deviation_pct": spike_pct,
                    "anomaly_score": round(float(row.z_score), 3),
                    "severity": "HIGH",
                    "description": f"Abnormal demand surge of +{spike_pct}%. Actual: {row.quantity} units vs rolling baseline {round(row.rolling_mean, 1)}.",
                    "action_item": "Validate inventory reserves and inspect for reseller/bot hoarding activity.",
                    "status": "OPEN"
                })

        return detected_alerts

    # ====================================================
    # Layer 2: Machine Learning Isolation Forest Screener
    # ====================================================
    def fit_isolation_forest(self, transactions_df: pd.DataFrame):
        """Fits the Isolation Forest on transaction amounts, quantities, and prices."""
        logger.info(f"Fitting Isolation Forest on {len(transactions_df)} transactions (contamination={self.contamination})...")
        X = transactions_df[self.TRANSACTION_FEATURES].copy()
        X_scaled = self.scaler.fit_transform(X)

        self.isolation_forest = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        logger.info("Isolation Forest fitted successfully.")

    def detect_transaction_anomalies(
        self,
        transactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Scores transactions using Isolation Forest and generates suspicious activity alerts.
        """
        if not self.is_fitted or self.isolation_forest is None:
            raise ValueError("Isolation Forest is not fitted. Call fit_isolation_forest() first.")

        df = transactions_df.copy()
        X = df[self.TRANSACTION_FEATURES]
        X_scaled = self.scaler.transform(X)

        # -1 = anomaly, 1 = normal
        predictions = self.isolation_forest.predict(X_scaled)
        # raw decision function: negative values indicate outliers
        scores = self.isolation_forest.decision_function(X_scaled)

        df["is_anomaly"] = predictions == -1
        # Normalize anomaly score from 0.0 (normal) to 1.0 (severe outlier)
        df["anomaly_score"] = np.round(1.0 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-6), 3)

        anomalies_df = df[df["is_anomaly"]].copy()
        logger.info(f"Detected {len(anomalies_df)} suspicious transaction anomalies.")

        alerts: List[Dict[str, Any]] = []
        for row in anomalies_df.itertuples():
            score = float(row.anomaly_score)
            severity = "CRITICAL" if score >= 0.85 else ("HIGH" if score >= 0.70 else "MEDIUM")

            alerts.append({
                "detected_at": getattr(row, "timestamp", datetime.utcnow()),
                "alert_type": "SUSPICIOUS_TRANSACTION",
                "entity_id": f"INV:{getattr(row, 'invoice_no', 'N/A')} | CUST:{getattr(row, 'customer_id', 'N/A')}",
                "metric_name": "basket_anomaly_score",
                "actual_value": float(row.total_amount),
                "expected_value": float(round(row.quantity * row.unit_price, 2)),
                "deviation_pct": round(score * 100, 1),
                "anomaly_score": score,
                "severity": severity,
                "description": f"Suspicious transaction detected! Order amount ${row.total_amount:,.2f} with quantity {row.quantity}x @ ${row.unit_price:.2f}/ea (Score: {score:.3f}).",
                "action_item": "Hold fulfillment. Trigger manual fraud screening and verify card billing address.",
                "status": "OPEN"
            })

        return df, alerts

    @staticmethod
    def persist_alerts(alerts: List[Dict[str, Any]]):
        """Persists detected anomaly alerts into the database."""
        if not alerts:
            return
        logger.info(f"Persisting {len(alerts)} anomaly alerts into SQL database...")
        with get_db_session() as session:
            for item in alerts:
                session.add(AnomalyAlert(
                    detected_at=item["detected_at"] if isinstance(item["detected_at"], datetime) else pd.to_datetime(item["detected_at"]),
                    alert_type=item["alert_type"],
                    entity_id=str(item["entity_id"]),
                    metric_name=item["metric_name"],
                    actual_value=float(item["actual_value"]),
                    expected_value=float(item["expected_value"]),
                    deviation_pct=float(item["deviation_pct"]),
                    anomaly_score=float(item["anomaly_score"]),
                    severity=item["severity"],
                    description=item["description"],
                    action_item=item["action_item"],
                    status=item.get("status", "OPEN")
                ))

    def save(self, filepath: Optional[str] = None):
        """Saves fitted Isolation Forest and Scaler."""
        if filepath is None:
            filepath = str(ARTIFACTS_DIR / "isolation_forest.joblib")
        joblib.dump({"model": self.isolation_forest, "scaler": self.scaler, "contamination": self.contamination}, filepath)
        logger.info(f"Saved Isolation Forest to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "AnomalyDetector":
        """Loads fitted Isolation Forest and Scaler."""
        if filepath is None:
            filepath = str(ARTIFACTS_DIR / "isolation_forest.joblib")
        payload = joblib.load(filepath)
        detector = cls(contamination=payload.get("contamination", 0.03))
        detector.isolation_forest = payload["model"]
        detector.scaler = payload["scaler"]
        detector.is_fitted = True
        logger.info(f"Loaded Isolation Forest from {filepath}")
        return detector
