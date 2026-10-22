from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.logger import logger
from src.database.connection import engine, get_db_session
from src.database.models import CleanedTransaction, DailyMetric, RawTransaction

class DataQualityAuditReport:
    """Encapsulates data cleaning statistics and quality metrics."""
    def __init__(self):
        self.total_raw_records: int = 0
        self.duplicates_removed: int = 0
        self.cancellations_handled: int = 0
        self.missing_customers_imputed: int = 0
        self.missing_descriptions_imputed: int = 0
        self.invalid_prices_removed: int = 0
        self.invalid_quantities_removed: int = 0
        self.final_cleaned_records: int = 0
        self.validation_checks: Dict[str, bool] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_raw_records": self.total_raw_records,
            "duplicates_removed": self.duplicates_removed,
            "cancellations_handled": self.cancellations_handled,
            "missing_customers_imputed": self.missing_customers_imputed,
            "missing_descriptions_imputed": self.missing_descriptions_imputed,
            "invalid_prices_removed": self.invalid_prices_removed,
            "invalid_quantities_removed": self.invalid_quantities_removed,
            "final_cleaned_records": self.final_cleaned_records,
            "clean_retention_rate_pct": round(
                (self.final_cleaned_records / max(1, self.total_raw_records)) * 100, 2
            ),
            "validation_checks": self.validation_checks
        }

class DataCleaner:
    """Robust data cleaning pipeline for e-commerce transactions."""

    PRICE_MIN = 0.50
    PRICE_MAX = 5000.00
    QUANTITY_MAX = 1000

    @classmethod
    def clean_batch(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityAuditReport]:
        """
        Cleans a batch of raw e-commerce records, performing schema validation,
        deduplication, cancellation separation, missing value imputation, and price sanity filtering.
        """
        report = DataQualityAuditReport()
        report.total_raw_records = len(df)
        logger.info(f"Starting data cleaning pipeline for {len(df)} raw records...")

        df_clean = df.copy()

        # Step 1: Remove exact duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["invoice_no", "stock_code", "invoice_date", "quantity"])
        report.duplicates_removed = initial_count - len(df_clean)
        logger.debug(f"Removed {report.duplicates_removed} duplicate records.")

        # Step 2: Handle Cancellations & Returns (Invoice starts with 'C' or quantity < 0)
        is_cancellation = (
            df_clean["invoice_no"].astype(str).str.startswith("C", na=False) |
            (pd.to_numeric(df_clean["quantity"], errors="coerce") <= 0)
        )
        report.cancellations_handled = int(is_cancellation.sum())
        df_clean = df_clean[~is_cancellation].copy()
        logger.debug(f"Identified and filtered {report.cancellations_handled} cancellations/returns.")

        # Step 3: Handle Missing Values
        # Customer ID: Impute missing customer IDs with guest placeholder
        missing_cust = df_clean["customer_id"].isna() | (df_clean["customer_id"] == "")
        report.missing_customers_imputed = int(missing_cust.sum())
        df_clean.loc[missing_cust, "customer_id"] = "CUST-GUEST-CHECKOUT"

        # Description / Product Name: Impute missing descriptions
        missing_desc = df_clean["description"].isna() | (df_clean["description"] == "")
        report.missing_descriptions_imputed = int(missing_desc.sum())
        df_clean.loc[missing_desc, "description"] = "Unspecified Retail Item"

        # Category: Default to 'General' if missing
        if "category" not in df_clean.columns:
            df_clean["category"] = "General"
        df_clean["category"] = df_clean["category"].fillna("General")

        # Step 4: Sanitize Numerical Fields (Quantities and Unit Prices)
        df_clean["quantity"] = pd.to_numeric(df_clean["quantity"], errors="coerce")
        df_clean["unit_price"] = pd.to_numeric(df_clean["unit_price"], errors="coerce")

        # Filter invalid unit prices ($0 or extreme outliers)
        invalid_prices = (
            df_clean["unit_price"].isna() |
            (df_clean["unit_price"] < cls.PRICE_MIN) |
            (df_clean["unit_price"] > cls.PRICE_MAX)
        )
        report.invalid_prices_removed = int(invalid_prices.sum())
        df_clean = df_clean[~invalid_prices].copy()

        # Filter invalid quantities (<=0 or extreme anomaly > QUANTITY_MAX)
        invalid_quantities = (
            df_clean["quantity"].isna() |
            (df_clean["quantity"] <= 0) |
            (df_clean["quantity"] > cls.QUANTITY_MAX)
        )
        report.invalid_quantities_removed = int(invalid_quantities.sum())
        df_clean = df_clean[~invalid_quantities].copy()

        # Step 5: Datetime Normalization
        df_clean["timestamp"] = pd.to_datetime(df_clean["invoice_date"], errors="coerce")
        df_clean = df_clean.dropna(subset=["timestamp"]).copy()

        # Step 6: Derived Feature Calculations
        df_clean["quantity"] = df_clean["quantity"].astype(int)
        df_clean["unit_price"] = df_clean["unit_price"].astype(float)
        df_clean["total_amount"] = np.round(df_clean["quantity"] * df_clean["unit_price"], 2)
        df_clean["is_weekend"] = df_clean["timestamp"].dt.weekday.isin([5, 6])
        df_clean["is_promo"] = df_clean.apply(
            lambda row: bool(row["unit_price"] < 50.0 and row["quantity"] >= 3), axis=1
        )

        # Standardize column naming
        df_clean["product_id"] = df_clean["stock_code"].astype(str)
        df_clean["product_name"] = df_clean["description"].astype(str)
        df_clean["country"] = df_clean["country"].fillna("Unknown").astype(str)

        # Create unique transaction ID if not present
        if "transaction_id" not in df_clean.columns:
            df_clean["transaction_id"] = [
                f"TXN-{row.invoice_no}-{i}" for i, row in enumerate(df_clean.itertuples())
            ]

        report.final_cleaned_records = len(df_clean)

        # Step 7: Data Contract & Validation Checks
        report.validation_checks = {
            "no_null_timestamps": bool(df_clean["timestamp"].isna().sum() == 0),
            "no_negative_quantities": bool((df_clean["quantity"] <= 0).sum() == 0),
            "no_zero_or_negative_prices": bool((df_clean["unit_price"] <= 0).sum() == 0),
            "no_null_customer_ids": bool(df_clean["customer_id"].isna().sum() == 0),
            "valid_total_amounts": bool((df_clean["total_amount"] < 0).sum() == 0)
        }

        logger.info(
            f"Data cleaning finished. Cleaned records: {report.final_cleaned_records} "
            f"({report.to_dict()['clean_retention_rate_pct']}% retention)."
        )
        return df_clean, report

    @classmethod
    def clean_single_record(cls, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Cleans and validates a single streaming transaction record."""
        df_single = pd.DataFrame([record])
        cleaned_df, report = cls.clean_batch(df_single)
        if cleaned_df.empty:
            return None
        return cleaned_df.iloc[0].to_dict()

    @classmethod
    def sync_to_database(cls, cleaned_df: pd.DataFrame):
        """Saves cleaned transactions and updates daily aggregated metrics in SQL database."""
        logger.info("Persisting cleaned transactions to SQL database...")
        # Persist transactions
        cleaned_df.to_sql(
            "cleaned_transactions",
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000
        )

        # Aggregate daily metrics for time-series forecasting & sales drop tracking
        daily_metrics = (
            cleaned_df.groupby([cleaned_df["timestamp"].dt.floor("D"), "product_id", "category"])
            .agg(
                total_quantity=("quantity", "sum"),
                total_revenue=("total_amount", "sum"),
                order_count=("invoice_no", "nunique"),
                avg_price=("unit_price", "mean"),
                is_promo=("is_promo", "max")
            )
            .reset_index()
            .rename(columns={"timestamp": "date"})
        )

        daily_metrics.to_sql(
            "daily_metrics",
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000
        )
        logger.info(f"Updated daily_metrics table with {len(daily_metrics)} daily aggregated records.")
