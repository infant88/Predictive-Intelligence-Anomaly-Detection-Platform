from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
from src.config import config, MLRUNS_DIR
from src.logger import logger

class MLflowTracker:
    """Manages experiment tracking, parameter logging, metric recording, and model registry via MLflow."""

    def __init__(self):
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self.experiment_name = config.mlflow_experiment_name
        self.client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        self._ensure_experiment()

    def _ensure_experiment(self):
        """Creates MLflow experiment if it does not already exist."""
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
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            # Log Parameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log Metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log Tags
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)
            mlflow.set_tag("model_name", model_name)

            run_id = run.info.run_id
            logger.info(f"MLflow logged run {run_id} for model '{model_name}'.")
            return run_id

    def get_recent_runs(self, max_runs: int = 15) -> List[Dict[str, Any]]:
        """Retrieves history of recent MLflow experiment runs for UI comparison."""
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if not exp:
                return []

            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=max_runs,
                order_by=["start_time DESC"]
            )

            results = []
            for r in runs:
                results.append({
                    "run_id": r.info.run_id,
                    "run_name": r.data.tags.get("mlflow.runName", "run"),
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "model_name": r.data.tags.get("model_name", "model")
                })
            return results
        except Exception as e:
            logger.error(f"Failed to fetch MLflow runs: {e}")
            return []

mlflow_tracker = MLflowTracker()
