import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.environ.get("VERCEL"))

if IS_VERCEL:
    DATA_DIR = Path("/tmp/data")
    ARTIFACTS_DIR = Path("/tmp/artifacts")
    MLRUNS_DIR = Path("/tmp/mlruns")
    LOGS_DIR = Path("/tmp/logs")
else:
    DATA_DIR = BASE_DIR / "data"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MLRUNS_DIR = BASE_DIR / "mlruns"
    LOGS_DIR = BASE_DIR / "logs"

for dir_path in [DATA_DIR, ARTIFACTS_DIR, MLRUNS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

DB_PATH = DATA_DIR / "platform.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

class PlatformConfig(BaseModel):
    app_name: str = "Predictive Intelligence & Anomaly Detection Platform"
    version: str = "2.0.0"
    debug: bool = True
    
    # Database
    database_url: str = DATABASE_URL
    
    # MLflow
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = "ecommerce_demand_and_anomalies"
    
    # Forecasting Settings
    forecast_horizon_days: int = 14
    model_type: str = "xgboost"  # 'xgboost' or 'random_forest'
    test_size_days: int = 30
    
    # Anomaly Detection Settings
    sales_drop_z_threshold: float = -2.0  # Z-score threshold for sudden sales drop
    contamination_rate: float = 0.03      # 3% expected transaction anomalies
    isolation_forest_estimators: int = 150
    
    # API & Dashboard
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501

config = PlatformConfig()
