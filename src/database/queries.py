from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine, get_db_session
from src.database.models import CleanedTransaction, DailyMetric, DemandForecast, AnomalyAlert
from src.logger import logger

class DatabaseRepository:
    """Analytical repository providing optimized SQL queries and transactional methods."""

    @staticmethod
    def get_kpis() -> Dict[str, Any]:
        """Calculates executive high-level business KPIs."""
        with get_db_session() as session:
            # Total revenue and volume
            sql = text("""
                SELECT 
                    COUNT(*) as total_orders,
                    COALESCE(SUM(total_amount), 0) as total_revenue,
                    COALESCE(SUM(quantity), 0) as total_units_sold,
                    COUNT(DISTINCT customer_id) as total_customers,
                    COUNT(DISTINCT product_id) as total_products
                FROM cleaned_transactions
            """)
            result = session.execute(sql).mappings().one()

            # Active open anomaly alerts
            alert_sql = text("""
                SELECT 
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical_alerts,
                    SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_alerts
                FROM anomaly_alerts
                WHERE status = 'OPEN'
            """)
            alert_res = session.execute(alert_sql).mappings().one()

            return {
                "total_orders": result["total_orders"],
                "total_revenue": round(float(result["total_revenue"]), 2),
                "total_units_sold": int(result["total_units_sold"]),
                "total_customers": result["total_customers"],
                "total_products": result["total_products"],
                "open_alerts": alert_res["total_alerts"] or 0,
                "critical_alerts": alert_res["critical_alerts"] or 0,
                "high_alerts": alert_res["high_alerts"] or 0
            }

    @staticmethod
    def get_daily_revenue_timeseries(days: int = 60) -> pd.DataFrame:
        """Retrieves daily aggregated sales and volume over time."""
        query = text("""
            SELECT 
                DATE(timestamp) as date,
                SUM(total_amount) as daily_revenue,
                SUM(quantity) as daily_quantity,
                COUNT(DISTINCT invoice_no) as order_count
            FROM cleaned_transactions
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp) ASC
        """)
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def get_product_daily_series(product_id: str) -> pd.DataFrame:
        """Gets daily sales series for a specific product."""
        query = text("""
            SELECT 
                DATE(timestamp) as date,
                product_id,
                category,
                SUM(quantity) as quantity,
                SUM(total_amount) as revenue,
                AVG(unit_price) as avg_price,
                MAX(is_promo) as is_promo
            FROM cleaned_transactions
            WHERE product_id = :product_id
            GROUP BY DATE(timestamp), product_id, category
            ORDER BY DATE(timestamp) ASC
        """)
        df = pd.read_sql(query, con=engine, params={"product_id": product_id})
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def get_top_products(limit: int = 10) -> pd.DataFrame:
        """Gets top performing products by total volume sold."""
        query = text("""
            SELECT 
                product_id,
                product_name,
                category,
                SUM(quantity) as total_quantity,
                SUM(total_amount) as total_revenue,
                COUNT(DISTINCT customer_id) as unique_buyers
            FROM cleaned_transactions
            GROUP BY product_id, product_name, category
            ORDER BY total_quantity DESC
            LIMIT :limit
        """)
        return pd.read_sql(query, con=engine, params={"limit": limit})

    @staticmethod
    def get_recent_alerts(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves anomaly alerts sorted by detection time."""
        with get_db_session() as session:
            base_query = session.query(AnomalyAlert)
            if status:
                base_query = base_query.filter(AnomalyAlert.status == status)
            alerts = base_query.order_by(AnomalyAlert.detected_at.desc()).limit(limit).all()

            return [
                {
                    "id": a.id,
                    "detected_at": a.detected_at.strftime("%Y-%m-%d %H:%M:%S") if a.detected_at else "",
                    "alert_type": a.alert_type,
                    "entity_id": a.entity_id,
                    "metric_name": a.metric_name,
                    "actual_value": round(a.actual_value, 2),
                    "expected_value": round(a.expected_value, 2),
                    "deviation_pct": round(a.deviation_pct, 2),
                    "anomaly_score": round(a.anomaly_score, 3),
                    "severity": a.severity,
                    "description": a.description,
                    "action_item": a.action_item,
                    "status": a.status
                }
                for a in alerts
            ]

    @staticmethod
    def get_latest_forecasts(product_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieves latest generated demand forecasts."""
        sql = """
            SELECT 
                forecast_created_at,
                target_date,
                product_id,
                category,
                predicted_quantity,
                lower_bound,
                upper_bound,
                model_name
            FROM demand_forecasts
        """
        params = {}
        if product_id:
            sql += " WHERE product_id = :product_id"
            params["product_id"] = product_id
        sql += " ORDER BY target_date ASC"

        df = pd.read_sql(text(sql), con=engine, params=params)
        if not df.empty:
            df["target_date"] = pd.to_datetime(df["target_date"])
        return df

    @staticmethod
    def run_arbitrary_query(sql_str: str) -> pd.DataFrame:
        """Safely executes a read-only SQL query for the interactive SQL explorer."""
        clean_sql = sql_str.strip()
        if not clean_sql.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted in the explorer.")
        return pd.read_sql(text(clean_sql), con=engine)
