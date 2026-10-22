import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import ARTIFACTS_DIR
from src.logger import logger
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.database.connection import get_db_session
from src.database.models import DemandForecast

class DemandForecaster:
    """Predicts future business demand using modern Gradient Boosted Trees and Random Forests."""

    def __init__(self, model_type: str = "xgboost"):
        if model_type.lower() == "xgboost" and not HAS_XGBOOST:
            logger.info("XGBoost not installed; defaulting to RandomForestRegressor.")
            self.model_type = "random_forest"
        else:
            self.model_type = model_type.lower()
        self.model = None
        self.residual_std: float = 1.0
        self.metrics: Dict[str, float] = {}
        self.feature_names: List[str] = FeatureEngineer.FEATURE_COLS
        self.trained_at: Optional[datetime] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Trains the demand forecasting model and calculates evaluation metrics."""
        logger.info(f"Training {self.model_type.upper()} Demand Forecaster on {len(X_train)} samples...")

        if self.model_type == "xgboost" and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            )

        self.model.fit(X_train, y_train)
        self.trained_at = datetime.utcnow()

        # Calculate Residuals & Uncertainty STD
        train_preds = self.model.predict(X_train)
        residuals = y_train.values - train_preds
        self.residual_std = float(np.std(residuals)) if len(residuals) > 0 else 1.0

        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
        train_mae = float(mean_absolute_error(y_train, train_preds))
        train_r2 = float(r2_score(y_train, train_preds))

        self.metrics = {
            "train_rmse": round(train_rmse, 3),
            "train_mae": round(train_mae, 3),
            "train_r2": round(train_r2, 3)
        }

        if X_test is not None and y_test is not None and len(X_test) > 0:
            test_preds = self.model.predict(X_test)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_mae = float(mean_absolute_error(y_test, test_preds))
            test_r2 = float(r2_score(y_test, test_preds))
            # WAPE (Weighted Absolute Percentage Error)
            total_actual = np.sum(np.abs(y_test))
            test_wape = float(np.sum(np.abs(y_test - test_preds)) / max(1.0, total_actual)) * 100

            self.metrics.update({
                "test_rmse": round(test_rmse, 3),
                "test_mae": round(test_mae, 3),
                "test_r2": round(test_r2, 3),
                "test_wape_pct": round(test_wape, 2)
            })
            logger.info(
                f"Evaluation Results -> Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, "
                f"Test R2: {test_r2:.3f}, WAPE: {test_wape:.1f}%"
            )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates point predictions for a feature matrix."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")
        return np.maximum(0, self.model.predict(X[self.feature_names]))

    def forecast_future_product_demand(
        self,
        historical_series: pd.DataFrame,
        horizon_days: int = 14,
        promo_discount_pct: float = 0.0,
        persist_db: bool = False
    ) -> pd.DataFrame:
        """
        Executes recursive multi-step forecasting for a specific product into the future.
        Supports What-If scenario simulation via promo_discount_pct.
        """
        if self.model is None:
            raise ValueError("Forecaster model is not trained.")

        df = historical_series.sort_values(by="date").copy()
        if len(df) < 21:
            raise ValueError(f"Need at least 21 days of history, got {len(df)}.")

        product_id = str(df["product_id"].iloc[0])
        category = str(df["category"].iloc[0]) if "category" in df.columns else "General"
        base_price = float(df["avg_price"].iloc[-1])
        effective_price = base_price * (1.0 - (promo_discount_pct / 100.0))
        is_promo = bool(promo_discount_pct > 5.0)

        quantities = list(df["quantity"].values)
        prices = list(df["avg_price"].values)
        last_date = pd.to_datetime(df["date"].iloc[-1])

        forecasts: List[Dict[str, Any]] = []

        for step in range(1, horizon_days + 1):
            target_date = last_date + timedelta(days=step)
            dow = target_date.weekday()
            dom = target_date.day
            month = target_date.month
            is_weekend = int(dow in (5, 6))

            # Build feature vector from recursive memory
            lag_1 = quantities[-1]
            lag_2 = quantities[-2]
            lag_7 = quantities[-7]
            lag_14 = quantities[-14]
            lag_21 = quantities[-21]

            recent_7 = quantities[-7:]
            rolling_mean_7 = float(np.mean(recent_7))
            rolling_std_7 = float(np.std(recent_7))
            rolling_max_7 = float(np.max(recent_7))
            rolling_min_7 = float(np.min(recent_7))

            recent_14 = quantities[-14:]
            rolling_mean_14 = float(np.mean(recent_14))

            recent_prices_7 = prices[-7:]
            rolling_price_7 = float(np.mean(recent_prices_7))
            price_ratio_7d = float(effective_price / (rolling_price_7 + 1e-6))

            feature_dict = {
                "lag_1": lag_1, "lag_2": lag_2, "lag_7": lag_7, "lag_14": lag_14, "lag_21": lag_21,
                "rolling_mean_7": rolling_mean_7, "rolling_std_7": rolling_std_7,
                "rolling_mean_14": rolling_mean_14,
                "rolling_max_7": rolling_max_7, "rolling_min_7": rolling_min_7,
                "day_of_week": dow, "day_of_month": dom, "month": month, "is_weekend": is_weekend,
                "avg_price": effective_price, "is_promo": int(is_promo), "price_ratio_7d": price_ratio_7d
            }

            feat_df = pd.DataFrame([feature_dict])
            pred_val = float(self.predict(feat_df)[0])

            # Apply promo boost if what-if promotion is simulated
            if promo_discount_pct > 0:
                promo_elasticity = 1.0 + (promo_discount_pct / 100.0) * 1.8
                pred_val *= promo_elasticity

            pred_val = round(max(0.0, pred_val), 1)

            # Prediction intervals (95% CI ~ 1.96 * std)
            horizon_uncertainty = self.residual_std * np.sqrt(1.0 + 0.05 * step)
            lower_bound = round(max(0.0, pred_val - 1.96 * horizon_uncertainty), 1)
            upper_bound = round(pred_val + 1.96 * horizon_uncertainty, 1)

            forecasts.append({
                "target_date": target_date,
                "product_id": product_id,
                "category": category,
                "predicted_quantity": pred_val,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model_name": self.model_type,
                "promo_discount_pct": promo_discount_pct,
                "is_promo": is_promo,
                "confidence": 0.95
            })

            # Append to recursive history for subsequent steps
            quantities.append(pred_val)
            prices.append(effective_price)

        forecast_df = pd.DataFrame(forecasts)

        if persist_db:
            with get_db_session() as session:
                for row in forecast_df.itertuples():
                    session.add(DemandForecast(
                        forecast_created_at=datetime.utcnow(),
                        target_date=pd.to_datetime(row.target_date),
                        product_id=row.product_id,
                        category=row.category,
                        predicted_quantity=row.predicted_quantity,
                        lower_bound=row.lower_bound,
                        upper_bound=row.upper_bound,
                        model_name=row.model_name,
                        confidence=row.confidence
                    ))
            logger.info(f"Persisted {len(forecast_df)} demand forecasts to database.")

        return forecast_df

    def save(self, filepath: Optional[str] = None):
        """Persists trained forecaster model and metadata."""
        if filepath is None:
            filepath = str(ARTIFACTS_DIR / f"{self.model_type}_forecaster.joblib")
        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "residual_std": self.residual_std,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "trained_at": self.trained_at
        }
        joblib.dump(payload, filepath)
        logger.info(f"Saved forecaster model to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "DemandForecaster":
        """Loads trained forecaster model from disk."""
        if filepath is None:
            filepath = str(ARTIFACTS_DIR / "xgboost_forecaster.joblib")
            if not os.path.exists(filepath):
                filepath = str(ARTIFACTS_DIR / "random_forest_forecaster.joblib")
        payload = joblib.load(filepath)
        forecaster = cls(model_type=payload["model_type"])
        forecaster.model = payload["model"]
        forecaster.residual_std = payload["residual_std"]
        forecaster.metrics = payload["metrics"]
        forecaster.feature_names = payload["feature_names"]
        forecaster.trained_at = payload.get("trained_at")
        logger.info(f"Loaded forecaster model from {filepath}")
        return forecaster
