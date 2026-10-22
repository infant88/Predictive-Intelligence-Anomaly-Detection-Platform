import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_tests():
    print("Running platform test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-p", "no:dash"]
    subprocess.run(cmd, cwd=BASE_DIR)

def run_pipeline():
    print("Running data cleaning, validation, and model training pipeline...")
    from src.database.connection import init_db
    from src.data_pipeline.generator import DataGenerator
    from src.data_pipeline.cleaner import DataCleaner
    from src.api.routes import train_platform_models

    init_db()
    raw = DataGenerator.generate_raw_ecommerce_data(days=90, include_dirty_data=True)
    clean, report = DataCleaner.clean_batch(raw)
    DataCleaner.sync_to_database(clean)
    train_platform_models()
    print("Pipeline and models completed successfully.")

def start_api():
    print("Starting FastAPI Backend at http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    subprocess.run(cmd, cwd=BASE_DIR)

def start_ui():
    print("Starting Streamlit Dashboard at http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501"]
    subprocess.run(cmd, cwd=BASE_DIR)

if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if arg == "test":
        run_tests()
    elif arg == "train" or arg == "pipeline":
        run_pipeline()
    elif arg == "api":
        start_api()
    elif arg == "ui" or arg == "streamlit":
        start_ui()
    else:
        print("""
Predictive Intelligence & Anomaly Detection Platform
===================================================
Commands:
  python run_platform.py test       - Run all unit tests with pytest
  python run_platform.py pipeline   - Ingest real-world data, clean, and train ML models
  python run_platform.py api        - Start FastAPI backend (http://127.0.0.1:8000)
  python run_platform.py ui         - Start Streamlit dashboard (http://localhost:8501)
        """)
