from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from src.config import config, DATA_DIR, MLRUNS_DIR
from src.logger import logger

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.info("MLflow not installed in this environment; using local fallback tracker.")

class MLflowTracker:
    """Manages experiment tracking, parameter logging, metric recording, and model registry via MLflow (with local fallback)."""

    def __init__(self):
        self.experiment_name = config.mlflow_experiment_name
        self.fallback_file = DATA_DIR / "local_runs.json"
        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
                self.client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
                self._ensure_experiment()
            except Exception as e:
                logger.warning(f"Could not connect to MLflow tracking server: {e}")
                self.client = None
        else:
            self.client = None

    def _ensure_experiment(self):
        """Creates MLflow experiment if it does not already exist."""
        if not HAS_MLFLOW or not self.client:
            return
        try:
            exp = self.client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                self.client.create_experiment(
                    name=self.experiment_name,
                    artifact_location=f"file:///{MLRUNS_DIR.as_posix()}"
                )
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow experiment initialization note: {e}")

    def log_training_run(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Logs a single model training run with parameters and metrics."""
        import uuid
        run_id = uuid.uuid4().hex[:12]

        if HAS_MLFLOW and self.client:
            try:
                mlflow.set_experiment(self.experiment_name)
                with mlflow.start_run(run_name=f"{model_name}_run") as run:
                    for k, v in params.items():
                        mlflow.log_param(k, v)
                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)
                    if tags:
                        for k, v in tags.items():
                            mlflow.set_tag(k, v)
                    mlflow.set_tag("model_name", model_name)
                    run_id = run.info.run_id
                logger.info(f"MLflow logged run {run_id} for model '{model_name}'.")
                return run_id
            except Exception as e:
                logger.warning(f"MLflow logging failed, falling back to local storage: {e}")

        # Local fallback persistence
        runs = self._load_fallback_runs()
        runs.insert(0, {
            "run_id": run_id,
            "run_name": f"{model_name}_run",
            "status": "FINISHED",
            "start_time": None,
            "metrics": metrics,
            "params": params,
            "model_name": model_name
        })
        self._save_fallback_runs(runs[:30])
        logger.info(f"Local tracker recorded run {run_id} for model '{model_name}'.")
        return run_id

    def _load_fallback_runs(self) -> List[Dict[str, Any]]:
        if self.fallback_file.exists():
            try:
                with open(self.fallback_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_fallback_runs(self, runs: List[Dict[str, Any]]):
        try:
            with open(self.fallback_file, "w") as f:
                json.dump(runs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist fallback runs: {e}")

    def get_recent_runs(self, max_runs: int = 15) -> List[Dict[str, Any]]:
        """Retrieves history of recent MLflow or local experiment runs."""
        if HAS_MLFLOW and self.client:
            try:
                exp = self.client.get_experiment_by_name(self.experiment_name)
                if exp:
                    runs = self.client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        max_results=max_runs,
                        order_by=["start_time DESC"]
                    )
                    return [
                        {
                            "run_id": r.info.run_id,
                            "run_name": r.data.tags.get("mlflow.runName", "run"),
                            "status": r.info.status,
                            "start_time": r.info.start_time,
                            "metrics": r.data.metrics,
                            "params": r.data.params,
                            "model_name": r.data.tags.get("model_name", "model")
                        }
                        for r in runs
                    ]
            except Exception as e:
                logger.warning(f"Could not retrieve MLflow runs: {e}")
        return self._load_fallback_runs()[:max_runs]
mlflow_tracker = MLflowTracker()
