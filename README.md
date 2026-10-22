# ⚡ Predictive Intelligence & Anomaly Detection Platform

An enterprise-grade Machine Learning platform that **predicts future business demand**, **detects unusual patterns and anomalies** (sudden sales crashes and fraudulent transactions), and **explains the key factors driving predictions** using Explainable AI (SHAP).

---

## 🏗️ Tech Stack

- **Core ML & Data**: Python 3.13, Pandas, Scikit-learn, XGBoost, SHAP
- **Database Layer**: SQLite & SQLAlchemy (ORM & Analytical SQL Repository)
- **API Framework**: FastAPI, Pydantic v2, Uvicorn
- **Interactive UI**: Streamlit, Plotly Express & Graph Objects
- **ML Governance**: MLflow v3 (SQLite tracking store, parameters, evaluation metrics)
- **Testing & Debugging**: Pytest, Custom structured logger with console & file rotation

---

## 🏛️ Platform Architecture

```
Predictive Intelligence & Anomaly Detection Platform/
├── src/
│   ├── config.py                 # Paths, MLflow tracking URI, thresholds
│   ├── logger.py                 # Structured logging & debugging setup
│   ├── database/
│   │   ├── connection.py         # SQLAlchemy engine & session manager
│   │   ├── models.py             # ORM: CleanedTransaction, DailyMetric, AnomalyAlert, DemandForecast
│   │   └── queries.py            # High-performance analytical SQL queries & KPIs
│   ├── data_pipeline/
│   │   ├── generator.py          # Real-world e-commerce data & live streaming simulator
│   │   ├── cleaner.py            # Data cleaning pipeline & automated quality audit
│   │   └── feature_engineering.py# Time-series lags, rolling windows, date/promo features
│   ├── models/
│   │   ├── forecaster.py         # XGBoost & Random Forest demand forecasting with uncertainty
│   │   ├── anomaly_detector.py   # Dual-layer: Z-Score Sales Drop + Isolation Forest Fraud Screener
│   │   ├── explainability.py     # SHAP TreeExplainer & Executive Business Narrative generator
│   │   └── mlflow_tracker.py     # MLflow logging, model registry & metrics tracking
│   ├── api/
│   │   ├── main.py               # FastAPI app, CORS, lifespan hooks
│   │   ├── schemas.py            # Pydantic v2 validation contracts
│   │   └── routes.py             # Endpoints for Ingestion, Forecast, Explain, Anomalies, MLflow
│   └── app/
│       └── streamlit_app.py      # Multi-page interactive Streamlit dashboard
├── tests/
│   ├── test_data_cleaner.py      # Deduplication, cancellations, imputation, price sanity tests
│   ├── test_feature_engineering.py# Lags, rolling stats, temporal split tests
│   ├── test_forecaster.py        # XGBoost training, recursive forecast, what-if uplift tests
│   ├── test_anomaly_detector.py  # Sales drop Z-score, Isolation Forest outlier tests
│   ├── test_explainability.py    # Global & local SHAP, executive narrative tests
│   └── test_api.py               # FastAPI endpoints, health, stream events, alerts tests
├── run_platform.py               # Unified CLI runner (test, pipeline, api, ui)
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Run All Unit Tests
Run the automated test suite with 100% passing coverage:
```bash
python run_platform.py test
```

### 2. Ingest Data & Train Models
Run the automated cleaning pipeline and train the models:
```bash
python run_platform.py pipeline
```

### 3. Launch FastAPI Backend
Start the high-performance REST API:
```bash
python run_platform.py api
```
Access interactive Swagger API Docs at: `http://127.0.0.1:8000/docs`

### 4. Launch Streamlit Executive Dashboard
Launch the interactive web application:
```bash
python run_platform.py ui
```
Open your browser at: `http://localhost:8501`

---

## 🎯 Key Capabilities

### 1. Demand Forecasting with What-If Simulation
- **Multi-Step Recursive Horizon**: Forecast demand 7, 14, 21, or 30 days ahead per product or category.
- **Uncertainty Bounds**: Generates 95% Confidence Intervals (`lower_bound` and `upper_bound`).
- **What-If Promotion Simulator**: Dynamically test the effect of promotional discounts (e.g. 15% discount) on demand uplift based on estimated price elasticities.

### 2. Dual-Layer Anomaly Detection
- **Time-Series Sales Drop Watchdog**: Uses rolling statistics, Z-score, and moving averages to identify sudden 50%-80% sales collapses (e.g., website checkout payment outages, inventory stockouts, or search de-indexation).
- **Machine Learning Fraud Screener**: Leverages Scikit-learn **Isolation Forest** to screen transaction-level amounts, unit prices, and quantities for bot orders, card testing, or price abuse.
- **Alert Prioritization**: Automatically classifies anomalies into `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW` with actionable remediation guidance.

### 3. Explainable AI (XAI) with SHAP
- **Global Feature Importance**: Identifies which factors most strongly drive sales across the catalog (`rolling_mean_7`, `lag_1`, `is_promo`, `day_of_week`).
- **Local Waterfall Attribution**: Explains any individual prediction by decomposing it into positive catalysts (green) and negative dampeners (red).
- **Executive Plain-English Narratives**: Translates mathematical SHAP values into readable business summaries for non-technical stakeholders.

### 4. Automated Data Cleaning & Integrity Contracts
- **Cancellation Separation**: Identifies `C`-prefix invoices and negative return quantities.
- **Missing Value Imputation**: Imputes customer IDs (`CUST-GUEST-CHECKOUT`) and default descriptions.
- **Deduplication**: Eliminates duplicate transaction logs.
- **Price & Quantity Sanity Filtering**: Removes zero-dollar transactions and extreme clerical errors ($99,999 typos).
- **Quality Audit Reporting**: Generates real-time audit metrics and contract validation checks.

### 5. MLflow Model Governance
- Tracks experiment runs, hyperparameter configurations, and evaluation metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Coefficient of Determination ($R^2$)
  - Weighted Absolute Percentage Error (WAPE)
