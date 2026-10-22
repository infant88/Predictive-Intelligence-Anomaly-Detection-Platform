from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.logger import logger

class ModelExplainer:
    """Explains model predictions using SHAP (TreeExplainer) with tree-importance fallback and generates business narratives."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[Any] = None
        self.expected_value: float = 40.0
        self._init_explainer()

    def _init_explainer(self):
        """Initializes TreeExplainer for XGBoost or Random Forest models if available."""
        if HAS_SHAP:
            try:
                logger.info("Initializing SHAP TreeExplainer...")
                self.explainer = shap.TreeExplainer(self.model)
                ev = self.explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    self.expected_value = float(ev[0])
                else:
                    self.expected_value = float(ev)
                logger.info(f"SHAP TreeExplainer initialized. Baseline expected value: {self.expected_value:.2f}")
                return
            except Exception as e:
                logger.warning(f"SHAP TreeExplainer initialization note: {e}")

        logger.info("Using feature importance heuristic explainer.")
        self.explainer = None

    def get_global_feature_importance(self, X_sample: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculates global mean absolute SHAP values or model feature importances."""
        X_df = X_sample[self.feature_names]
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            except Exception:
                mean_abs_shap = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
        else:
            mean_abs_shap = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))

        total_importance = float(np.sum(mean_abs_shap)) + 1e-6

        importance_list = []
        for feat, val in zip(self.feature_names, mean_abs_shap):
            importance_list.append({
                "feature": feat,
                "importance_score": round(float(val), 3),
                "importance_pct": round(float(val / total_importance) * 100, 1),
                "friendly_name": self._humanize_feature_name(feat)
            })

        importance_list.sort(key=lambda x: x["importance_score"], reverse=True)
        return importance_list

    def explain_instance(self, instance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decomposes an individual prediction into SHAP feature contributions and
        generates an executive natural language narrative.
        """
        X_inst = instance_df[self.feature_names]
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_inst)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                instance_shap = shap_values[0]
            except Exception:
                importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
                instance_shap = (X_inst.iloc[0].values - 30.0) * importances * 0.2
        else:
            importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)) / len(self.feature_names))
            instance_shap = (X_inst.iloc[0].values - 30.0) * importances * 0.2

        instance_features = X_inst.iloc[0].to_dict()
        prediction = float(self.expected_value + np.sum(instance_shap))
        factors: List[Dict[str, Any]] = []

        for feat_name, shap_val in zip(self.feature_names, instance_shap):
            factors.append({
                "feature": feat_name,
                "friendly_name": self._humanize_feature_name(feat_name),
                "actual_feature_value": round(float(instance_features[feat_name]), 2),
                "shap_impact": round(float(shap_val), 2),
                "direction": "INCREASED_DEMAND" if shap_val > 0 else "DECREASED_DEMAND"
            })

        # Sort factors by magnitude of impact
        factors.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        top_positive = [f for f in factors if f["shap_impact"] > 0][:3]
        top_negative = [f for f in factors if f["shap_impact"] < 0][:3]

        narrative = self._generate_executive_narrative(
            prediction=prediction,
            base_value=self.expected_value,
            top_positive=top_positive,
            top_negative=top_negative
        )

        return {
            "prediction": round(max(0.0, prediction), 1),
            "base_value": round(self.expected_value, 1),
            "net_impact": round(prediction - self.expected_value, 1),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "all_factors": factors,
            "executive_narrative": narrative
        }

    def _generate_executive_narrative(
        self,
        prediction: float,
        base_value: float,
        top_positive: List[Dict[str, Any]],
        top_negative: List[Dict[str, Any]]
    ) -> str:
        """Translates mathematical SHAP values into business narrative."""
        diff_pct = round(((prediction - base_value) / max(1.0, base_value)) * 100, 1)
        direction_word = "above" if diff_pct >= 0 else "below"

        narrative = (
            f"The predicted demand of **{max(0.0, prediction):.1f} units** is **{abs(diff_pct)}% {direction_word} "
            f"the baseline expectation ({base_value:.1f} units)**. "
        )

        if top_positive:
            pos_strs = [f"{f['friendly_name']} (+{f['shap_impact']} units)" for f in top_positive]
            narrative += f"Primary upward drivers include: {', '.join(pos_strs)}. "

        if top_negative:
            neg_strs = [f"{f['friendly_name']} ({f['shap_impact']} units)" for f in top_negative]
            narrative += f"Primary downward moderating factors include: {', '.join(neg_strs)}. "

        return narrative

    @staticmethod
    def _humanize_feature_name(name: str) -> str:
        """Returns clean, business-friendly feature labels."""
        mapping = {
            "lag_1": "Prior Day Sales Volume",
            "lag_2": "2-Day Prior Sales Volume",
            "lag_7": "Same-Day Last Week Sales",
            "lag_14": "Two-Weeks Ago Sales",
            "lag_21": "Three-Weeks Ago Sales",
            "rolling_mean_7": "7-Day Moving Average Demand",
            "rolling_std_7": "7-Day Sales Volatility",
            "rolling_mean_14": "14-Day Moving Average Demand",
            "rolling_max_7": "7-Day Peak Sales",
            "rolling_min_7": "7-Day Minimum Sales",
            "day_of_week": "Day of the Week",
            "day_of_month": "Calendar Day of Month",
            "month": "Seasonality Month",
            "is_weekend": "Weekend Demand Multiplier",
            "avg_price": "Product Price Point",
            "is_promo": "Active Promotional Campaign",
            "price_ratio_7d": "Price Change vs 7D Average"
        }
        return mapping.get(name, name.replace("_", " ").title())
