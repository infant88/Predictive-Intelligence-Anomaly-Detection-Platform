import pytest
import pandas as pd
import numpy as np
from src.data_pipeline.cleaner import DataCleaner

def test_data_cleaner_removes_duplicates():
    raw_data = pd.DataFrame([
        {"invoice_no": "INV-100", "stock_code": "PROD-1", "description": "Item 1", "quantity": 2, "invoice_date": "2026-01-01 10:00:00", "unit_price": 20.0, "customer_id": "C-1", "country": "UK", "category": "Tech"},
        {"invoice_no": "INV-100", "stock_code": "PROD-1", "description": "Item 1", "quantity": 2, "invoice_date": "2026-01-01 10:00:00", "unit_price": 20.0, "customer_id": "C-1", "country": "UK", "category": "Tech"}
    ])
    cleaned_df, report = DataCleaner.clean_batch(raw_data)
    assert len(cleaned_df) == 1
    assert report.duplicates_removed == 1

def test_data_cleaner_filters_cancellations():
    raw_data = pd.DataFrame([
        {"invoice_no": "INV-101", "stock_code": "PROD-1", "description": "Valid Item", "quantity": 3, "invoice_date": "2026-01-01 10:00:00", "unit_price": 15.0, "customer_id": "C-1", "country": "UK"},
        {"invoice_no": "C-INV-102", "stock_code": "PROD-1", "description": "Cancelled Item", "quantity": -3, "invoice_date": "2026-01-01 11:00:00", "unit_price": 15.0, "customer_id": "C-1", "country": "UK"}
    ])
    cleaned_df, report = DataCleaner.clean_batch(raw_data)
    assert len(cleaned_df) == 1
    assert report.cancellations_handled == 1
    assert cleaned_df.iloc[0]["invoice_no"] == "INV-101"

def test_data_cleaner_imputes_missing_customers_and_descriptions():
    raw_data = pd.DataFrame([
        {"invoice_no": "INV-103", "stock_code": "PROD-2", "description": None, "quantity": 1, "invoice_date": "2026-01-01 12:00:00", "unit_price": 45.0, "customer_id": np.nan, "country": "US"}
    ])
    cleaned_df, report = DataCleaner.clean_batch(raw_data)
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["customer_id"] == "CUST-GUEST-CHECKOUT"
    assert cleaned_df.iloc[0]["product_name"] == "Unspecified Retail Item"
    assert report.missing_customers_imputed == 1

def test_data_cleaner_removes_invalid_prices():
    raw_data = pd.DataFrame([
        {"invoice_no": "INV-104", "stock_code": "PROD-3", "description": "Valid", "quantity": 1, "invoice_date": "2026-01-01 13:00:00", "unit_price": 25.0, "customer_id": "C-3", "country": "US"},
        {"invoice_no": "INV-105", "stock_code": "PROD-3", "description": "Zero Price", "quantity": 1, "invoice_date": "2026-01-01 13:00:00", "unit_price": 0.0, "customer_id": "C-3", "country": "US"},
        {"invoice_no": "INV-106", "stock_code": "PROD-3", "description": "Extreme Error", "quantity": 1, "invoice_date": "2026-01-01 13:00:00", "unit_price": 99999.0, "customer_id": "C-3", "country": "US"}
    ])
    cleaned_df, report = DataCleaner.clean_batch(raw_data)
    assert len(cleaned_df) == 1
    assert report.invalid_prices_removed == 2
    assert cleaned_df.iloc[0]["invoice_no"] == "INV-104"

def test_data_cleaner_validation_checks_passed():
    raw_data = pd.DataFrame([
        {"invoice_no": "INV-107", "stock_code": "PROD-4", "description": "Good Item", "quantity": 5, "invoice_date": "2026-01-02 14:00:00", "unit_price": 10.0, "customer_id": "C-4", "country": "DE", "category": "Home"}
    ])
    cleaned_df, report = DataCleaner.clean_batch(raw_data)
    assert all(report.validation_checks.values())
    assert cleaned_df.iloc[0]["total_amount"] == 50.0
