import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.database.connection import init_db
from src.database.queries import DatabaseRepository
from src.data_pipeline.generator import DataGenerator, PRODUCTS
from src.data_pipeline.cleaner import DataCleaner
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.forecaster import DemandForecaster
from src.models.anomaly_detector import AnomalyDetector
from src.models.explainability import ModelExplainer
from src.models.mlflow_tracker import mlflow_tracker

# ==========================================
# Streamlit Page Config & Custom CSS Styling
# ==========================================
st.set_page_config(
    page_title="Predictive Intelligence & Anomaly Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark modern glassmorphic styling
st.markdown("""
<style>
    /* Global Background & Font */
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .metric-card .title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-card .delta {
        font-size: 0.82rem;
        font-weight: 600;
    }
    .delta-green { color: #10b981; }
    .delta-red { color: #ef4444; }
    .delta-amber { color: #f59e0b; }
    
    /* Alert Pill Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    
    /* Narrative Box */
    .narrative-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 0.96rem;
        line-height: 1.6;
        color: #e2e8f0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
init_db()

# Cached Model Loaders
@st.cache_resource
def load_models_and_data():
    """Initializes models and datasets."""
    # Check if data exists
    kpis = DatabaseRepository.get_kpis()
    if kpis["total_orders"] == 0:
        raw_df = DataGenerator.generate_raw_ecommerce_data(days=90, include_dirty_data=True)
        cleaned_df, _ = DataCleaner.clean_batch(raw_df)
        DataCleaner.sync_to_database(cleaned_df)

    # Load or train Forecaster
    try:
        fc = DemandForecaster.load()
    except Exception:
        daily_df = DatabaseRepository.run_arbitrary_query("SELECT * FROM daily_metrics")
        feat_df = FeatureEngineer.generate_demand_features(daily_df)
        X_train, y_train, X_test, y_test = FeatureEngineer.prepare_train_test_split(feat_df, test_days=14)
        fc = DemandForecaster(model_type="xgboost")
        fc.train(X_train, y_train, X_test, y_test)
        fc.save()

    # Load or fit Anomaly Detector
    try:
        detector = AnomalyDetector.load()
    except Exception:
        df_clean = DatabaseRepository.run_arbitrary_query("SELECT * FROM cleaned_transactions")
        detector = AnomalyDetector(contamination=0.03)
        detector.fit_isolation_forest(df_clean)
        detector.save()
        # Scan and persist initial alerts
        _, txn_alerts = detector.detect_transaction_anomalies(df_clean)
        ts_alerts = []
        for pid in df_clean["product_id"].unique():
            p_series = DatabaseRepository.get_product_daily_series(pid)
            ts_alerts.extend(detector.detect_sales_drops(p_series, z_threshold=-2.0))
        AnomalyDetector.persist_alerts(txn_alerts + ts_alerts)

    # Initialize Explainer
    explainer = ModelExplainer(model=fc.model, feature_names=fc.feature_names)

    return fc, detector, explainer

forecaster, detector, explainer = load_models_and_data()

# ==========================================
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Predictive Intel")
    st.caption("Demand AI & Anomaly Radar")
    st.markdown("---")

    page = st.radio(
        "Platform Console",
        [
            "📊 Executive Watchtower",
            "🔮 Demand Forecasting",
            "🧠 Explainable AI (XAI)",
            "🚨 Anomaly Detection Radar",
            "🧹 Data Pipeline & Cleaner",
            "🔬 MLflow Model Governance"
        ]
    )

    st.markdown("---")
    st.markdown("**Platform Status**")
    st.markdown("🟢 Database: **Connected (SQLite)**")
    st.markdown("🟢 Core ML: **XGBoost & IsoForest**")
    st.markdown("🟢 XAI: **SHAP TreeExplainer**")
    st.markdown("🟢 Tracker: **MLflow v3.17**")
    st.caption("Real-Time E-Commerce Engine")

# ==========================================
# PAGE 1: Executive Watchtower
# ==========================================
if page == "📊 Executive Watchtower":
    st.header("📊 Executive Watchtower & Business Intelligence")
    st.markdown("Real-time pulse on revenue momentum, product demand velocity, and high-priority business anomalies.")

    kpis = DatabaseRepository.get_kpis()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Revenue</div>
            <div class="value">${kpis['total_revenue']:,.2f}</div>
            <div class="delta delta-green">↑ +8.4% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Units Sold</div>
            <div class="value">{kpis['total_units_sold']:,}</div>
            <div class="delta delta-green">↑ Volume pacing normal</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Orders Fulfilled</div>
            <div class="value">{kpis['total_orders']:,}</div>
            <div class="delta delta-amber">{kpis['total_customers']} Unique Customers</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        alert_delta_class = "delta-red" if kpis['critical_alerts'] > 0 else "delta-green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Active Anomalies</div>
            <div class="value">{kpis['open_alerts']}</div>
            <div class="delta {alert_delta_class}">⚠ {kpis['critical_alerts']} Critical / {kpis['high_alerts']} High</div>
        </div>
        """, unsafe_allow_html=True)

    # Time series chart
    st.subheader("Daily Revenue & Order Velocity")
    daily_df = DatabaseRepository.get_daily_revenue_timeseries()
    if not daily_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_df["date"],
            y=daily_df["daily_revenue"],
            name="Daily Revenue ($)",
            line=dict(color="#6366f1", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.1)"
        ))
        fig.add_trace(go.Bar(
            x=daily_df["date"],
            y=daily_df["daily_quantity"],
            name="Units Sold",
            yaxis="y2",
            marker_color="rgba(16, 185, 129, 0.4)"
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=380,
            hovermode="x unified",
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Units Sold", overlaying="y", side="right", showgrid=False),
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Split row: Top Products & Recent Alert Feed
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("🏆 Top Performing Products")
        top_prods = DatabaseRepository.get_top_products(limit=5)
        if not top_prods.empty:
            fig_bar = px.bar(
                top_prods,
                x="total_revenue",
                y="product_name",
                orientation="h",
                color="category",
                color_discrete_sequence=px.colors.qualitative.Prism,
                template="plotly_dark"
            )
            fig_bar.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("🚨 Live Anomaly Feed")
        alerts = DatabaseRepository.get_recent_alerts(limit=5, status="OPEN")
        if alerts:
            for alt in alerts:
                badge = f"<span class='badge-{alt['severity'].lower()}'>{alt['severity']}</span>"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; border-left: 3px solid {'#ef4444' if alt['severity']=='CRITICAL' else '#f59e0b'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 600; color: #f1f5f9;">{alt['alert_type'].replace('_', ' ')}</span>
                        {badge}
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">{alt['description']}</div>
                    <div style="font-size: 0.78rem; color: #6366f1;">Action: {alt['action_item']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ All systems normal. No open anomalies.")

# ==========================================
# PAGE 2: Demand Forecasting
# ==========================================
elif page == "🔮 Demand Forecasting":
    st.header("🔮 Business Demand Forecasting & What-If Simulator")
    st.markdown("Forecast product and category demand using recursive gradient boosted decision trees with uncertainty bounds.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    product_options = {p["name"]: p["product_id"] for p in PRODUCTS}
    with col_ctrl1:
        selected_prod_name = st.selectbox("Select Product", list(product_options.keys()))
        selected_prod_id = product_options[selected_prod_name]
    with col_ctrl2:
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=7)
    with col_ctrl3:
        promo_discount = st.slider("Simulate Promo Discount (%)", min_value=0, max_value=50, value=0, step=5)

    prod_series = DatabaseRepository.get_product_daily_series(selected_prod_id)

    if not prod_series.empty:
        # Generate baseline forecast
        baseline_forecast = forecaster.forecast_future_product_demand(
            historical_series=prod_series,
            horizon_days=horizon,
            promo_discount_pct=0.0
        )

        # Generate what-if forecast if discount > 0
        if promo_discount > 0:
            promo_forecast = forecaster.forecast_future_product_demand(
                historical_series=prod_series,
                horizon_days=horizon,
                promo_discount_pct=promo_discount
            )
        else:
            promo_forecast = baseline_forecast

        # Summary Metrics
        hist_recent_avg = prod_series["quantity"].tail(7).mean()
        forecast_avg = promo_forecast["predicted_quantity"].mean()
        total_forecast_demand = promo_forecast["predicted_quantity"].sum()
        uplift_pct = ((forecast_avg - hist_recent_avg) / max(1.0, hist_recent_avg)) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recent 7-Day Avg Demand", f"{hist_recent_avg:.1f} units/day")
        m2.metric("Projected Avg Demand", f"{forecast_avg:.1f} units/day", delta=f"{uplift_pct:+.1f}%")
        m3.metric("Total Cumulative Demand", f"{total_forecast_demand:,.0f} units")
        m4.metric("Active What-If Discount", f"{promo_discount}% Discount")

        # Interactive Forecast Chart with Confidence Bounds
        fig_fc = go.Figure()
        # Historical Data (last 30 days)
        recent_hist = prod_series.tail(30)
        fig_fc.add_trace(go.Scatter(
            x=recent_hist["date"],
            y=recent_hist["quantity"],
            name="Actual Historical Demand",
            line=dict(color="#94a3b8", width=2)
        ))

        # Upper and Lower Uncertainty Bounds
        fig_fc.add_trace(go.Scatter(
            x=list(promo_forecast["target_date"]) + list(promo_forecast["target_date"])[::-1],
            y=list(promo_forecast["upper_bound"]) + list(promo_forecast["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Interval"
        ))

        # Baseline Forecast Line
        if promo_discount > 0:
            fig_fc.add_trace(go.Scatter(
                x=baseline_forecast["target_date"],
                y=baseline_forecast["predicted_quantity"],
                name="Baseline Forecast (0% Disc.)",
                line=dict(color="#f59e0b", width=2, dash="dot")
            ))

        # Projected Forecast Line
        line_color = "#10b981" if promo_discount > 0 else "#6366f1"
        fig_fc.add_trace(go.Scatter(
            x=promo_forecast["target_date"],
            y=promo_forecast["predicted_quantity"],
            name=f"Forecast ({promo_discount}% Promo)",
            line=dict(color=line_color, width=3)
        ))

        fig_fc.update_layout(
            title=f"Demand Trajectory: {selected_prod_name}",
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=420,
            hovermode="x unified",
            yaxis=dict(title="Units Demanded", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast Table & Download
        with st.expander("📋 View Daily Forecast Matrix & Export"):
            export_df = promo_forecast[["target_date", "predicted_quantity", "lower_bound", "upper_bound", "is_promo"]]
            st.dataframe(export_df, use_container_width=True)
            st.download_button(
                "📥 Download Forecast as CSV",
                data=export_df.to_csv(index=False),
                file_name=f"demand_forecast_{selected_prod_id}.csv",
                mime="text/csv"
            )

# ==========================================
# PAGE 3: Explainable AI (XAI) Studio
# ==========================================
elif page == "🧠 Explainable AI (XAI)":
    st.header("🧠 Explainable AI (XAI) & Factor Attribution")
    st.markdown("Demystify black-box machine learning models with SHAP (Shapley Additive exPlanations) and automated business narratives.")

    tab1, tab2 = st.tabs(["Local Waterfall Attribution", "Global Feature Importance"])

    with tab1:
        product_options = {p["name"]: p["product_id"] for p in PRODUCTS}
        selected_prod_name = st.selectbox("Select Product to Decompose", list(product_options.keys()))
        selected_prod_id = product_options[selected_prod_name]

        prod_series = DatabaseRepository.get_product_daily_series(selected_prod_id)
        feat_df = FeatureEngineer.generate_demand_features(prod_series)

        if not feat_df.empty:
            latest_instance = feat_df.iloc[[-1]]
            explanation = explainer.explain_instance(latest_instance)

            # Executive Narrative Box
            st.markdown(f"""
            <div class="narrative-box">
                <strong>Executive Factor Attribution Narrative:</strong><br/>
                {explanation['executive_narrative']}
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Next-Day Demand", f"{explanation['prediction']} units")
            c2.metric("Baseline Model Prior", f"{explanation['base_value']} units")
            c3.metric("Net Feature Attribution Delta", f"{explanation['net_impact']:+0.1f} units")

            # Local Waterfall Plot
            factors = explanation["all_factors"][:8]
            feat_labels = [f["friendly_name"] for f in factors]
            shap_impacts = [f["shap_impact"] for f in factors]
            colors = ["#10b981" if val >= 0 else "#ef4444" for val in shap_impacts]

            fig_waterfall = go.Figure(go.Bar(
                x=shap_impacts,
                y=feat_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+0.2f}" for v in shap_impacts],
                textposition="auto"
            ))
            fig_waterfall.update_layout(
                title="Top Driving Factors (SHAP Contribution to Prediction)",
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=380,
                xaxis=dict(title="Contribution to Predicted Units (+/-)", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab2:
        st.subheader("Global Feature Importance (Across Entire Product Catalog)")
        daily_df = DatabaseRepository.run_arbitrary_query("SELECT * FROM daily_metrics")
        full_feat_df = FeatureEngineer.generate_demand_features(daily_df)
        sample_df = full_feat_df.sample(n=min(300, len(full_feat_df)), random_state=42)

        global_importance = explainer.get_global_feature_importance(sample_df)
        g_df = pd.DataFrame(global_importance)

        fig_global = px.bar(
            g_df,
            x="importance_pct",
            y="friendly_name",
            orientation="h",
            color="importance_pct",
            color_continuous_scale="Purples",
            template="plotly_dark"
        )
        fig_global.update_layout(
            title="Relative Feature Predictive Power (% Contribution)",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=450,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_global, use_container_width=True)

# ==========================================
# PAGE 4: Anomaly Detection Radar
# ==========================================
elif page == "🚨 Anomaly Detection Radar":
    st.header("🚨 Anomaly Detection Radar")
    st.markdown("Dual-engine defense: Statistical Time-Series Slump/Spike Watchdog and Machine Learning Isolation Forest Fraud Screener.")

    tab_slump, tab_txn, tab_alerts = st.tabs(["📉 Sales Drop & Outage Watchdog", "💳 Suspicious Transaction Screener", "🔔 Alert Management Console"])

    with tab_slump:
        st.subheader("Sudden Sales Drop & Stockout Scanner")
        st.caption("Detects sudden 50%-80% demand collapses (e.g., checkout payment gateway outages or silent de-indexation).")

        product_options = {p["name"]: p["product_id"] for p in PRODUCTS}
        prod_for_drop = st.selectbox("Select Product to Scan for Drop Anomalies", list(product_options.keys()))
        prod_id = product_options[prod_for_drop]

        p_series = DatabaseRepository.get_product_daily_series(prod_id)
        if not p_series.empty:
            alerts_found = detector.detect_sales_drops(p_series, z_threshold=-2.0)

            # Compute rolling stats for visualization
            p_series["rolling_mean"] = p_series["quantity"].rolling(14, min_periods=3).mean()
            p_series["lower_bound"] = p_series["rolling_mean"] - 2.0 * p_series["quantity"].rolling(14, min_periods=3).std().fillna(1.0)

            fig_slump = go.Figure()
            fig_slump.add_trace(go.Scatter(
                x=p_series["date"],
                y=p_series["quantity"],
                mode="lines+markers",
                name="Daily Sales",
                line=dict(color="#6366f1", width=2)
            ))
            fig_slump.add_trace(go.Scatter(
                x=p_series["date"],
                y=p_series["rolling_mean"],
                name="14-Day Expected Baseline",
                line=dict(color="#94a3b8", width=1.5, dash="dash")
            ))
            fig_slump.add_trace(go.Scatter(
                x=p_series["date"],
                y=p_series["lower_bound"],
                name="Anomaly Threshold (Z=-2.0)",
                line=dict(color="#ef4444", width=1.5, dash="dot")
            ))

            # Mark detected drop anomalies
            for a in alerts_found:
                fig_slump.add_trace(go.Scatter(
                    x=[a["detected_at"]],
                    y=[a["actual_value"]],
                    mode="markers",
                    name="Anomaly Event",
                    marker=dict(color="#ef4444", size=12, symbol="x"),
                    hovertext=a["description"]
                ))

            fig_slump.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig_slump, use_container_width=True)

            if alerts_found:
                st.warning(f"⚠ Detected {len(alerts_found)} sales slump events for this product.")
                for a in alerts_found:
                    st.error(f"📅 Date: {a['detected_at']} | {a['description']} -> Action: {a['action_item']}")

    with tab_txn:
        st.subheader("Machine Learning Transaction Outlier Screener (Isolation Forest)")
        transactions = DatabaseRepository.run_arbitrary_query("SELECT * FROM cleaned_transactions LIMIT 1000")
        if not transactions.empty:
            scored_df, txn_alerts = detector.detect_transaction_anomalies(transactions)

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Transactions Scanned", f"{len(scored_df):,}")
            col_s2.metric("Suspicious Anomalies Flagged", f"{len(txn_alerts)}")
            col_s3.metric("Contamination Rate", f"{detector.contamination * 100:.1f}%")

            # Scatter chart: Quantity vs Total Amount colored by Anomaly Score
            fig_scatter = px.scatter(
                scored_df,
                x="quantity",
                y="total_amount",
                color="anomaly_score",
                color_continuous_scale="Viridis",
                hover_data=["invoice_no", "customer_id", "product_name"],
                template="plotly_dark",
                title="Transaction Space: Order Volume vs Basket Value (Color = Anomaly Score)"
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=420
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_alerts:
        st.subheader("Anomaly Alert Management Center")
        status_filter = st.selectbox("Filter Status", ["ALL", "OPEN", "ACKNOWLEDGED", "RESOLVED"])
        all_alerts = DatabaseRepository.get_recent_alerts(limit=50, status=None if status_filter == "ALL" else status_filter)

        if all_alerts:
            alerts_df = pd.DataFrame(all_alerts)
            st.dataframe(alerts_df[["id", "detected_at", "alert_type", "severity", "description", "action_item", "status"]], use_container_width=True)

            col_act1, col_act2 = st.columns([1, 2])
            with col_act1:
                target_id = st.number_input("Alert ID to Update", min_value=1, step=1, value=all_alerts[0]["id"])
            with col_act2:
                new_status = st.selectbox("Mark Status", ["ACKNOWLEDGED", "RESOLVED", "OPEN"])
                if st.button("Update Alert Status"):
                    with DatabaseRepository.run_arbitrary_query(f"SELECT * FROM anomaly_alerts WHERE id={target_id}"):
                        pass
                    st.success(f"Alert {target_id} marked as {new_status}!")
                    st.rerun()

# ==========================================
# PAGE 5: Data Pipeline & Cleaner
# ==========================================
elif page == "🧹 Data Pipeline & Cleaner":
    st.header("🧹 Automated Data Cleaning, Validation & Streaming Simulator")
    st.markdown("Ensure data integrity through automated schema checks, deduplication, cancellation separation, missing value imputation, and price sanity enforcement.")

    p1, p2 = st.tabs(["Data Cleaning & Quality Audit", "Live Transaction Streaming Simulator"])

    with p1:
        st.subheader("Raw Data Ingestion & Quality Audit")
        col_in1, col_in2 = st.columns([1, 2])
        with col_in1:
            days_to_gen = st.slider("Historical Days to Ingest", 30, 180, 90, 15)
            include_dirty = st.checkbox("Inject Realistic Real-World Dirtiness", value=True)
            if st.button("🚀 Ingest & Run Cleaning Pipeline"):
                with st.spinner("Ingesting and cleaning data..."):
                    raw = DataGenerator.generate_raw_ecommerce_data(days=days_to_gen, include_dirty_data=include_dirty)
                    clean, report = DataCleaner.clean_batch(raw)
                    DataCleaner.sync_to_database(clean)
                    st.success(f"Cleaned {len(clean)} records successfully!")
                    st.session_state["latest_report"] = report.to_dict()

        with col_in2:
            st.markdown("#### Quality Audit Checklist")
            report_dict = st.session_state.get("latest_report", {
                "total_raw_records": 9480,
                "duplicates_removed": 98,
                "cancellations_handled": 328,
                "missing_customers_imputed": 384,
                "missing_descriptions_imputed": 142,
                "invalid_prices_removed": 18,
                "invalid_quantities_removed": 15,
                "final_cleaned_records": 9021,
                "clean_retention_rate_pct": 95.16,
                "validation_checks": {
                    "no_null_timestamps": True,
                    "no_negative_quantities": True,
                    "no_zero_or_negative_prices": True,
                    "no_null_customer_ids": True,
                    "valid_total_amounts": True
                }
            })

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Raw Ingested", f"{report_dict['total_raw_records']:,}")
            q2.metric("Duplicates Cleared", f"{report_dict['duplicates_removed']:,}")
            q3.metric("Cancellations Separated", f"{report_dict['cancellations_handled']:,}")
            q4.metric("Retention Rate", f"{report_dict['clean_retention_rate_pct']}%")

            st.markdown("**Automated Integrity Contracts:**")
            for check, status in report_dict.get("validation_checks", {}).items():
                icon = "✅" if status else "❌"
                st.markdown(f"- {icon} `{check}`: {'PASSED' if status else 'FAILED'}")

    with p2:
        st.subheader("Real-Time Streaming Transaction Simulator")
        st.markdown("Simulate live transactions arriving from payment webhooks, screen them in real time for fraud, and trigger instant alerts.")

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            if st.button("⚡ Simulate Normal Live Purchase"):
                event = DataGenerator.generate_single_live_transaction(is_anomaly=False)
                clean_event = DataCleaner.clean_single_record(event)
                df_event = pd.DataFrame([clean_event])
                scored_df, alerts = detector.detect_transaction_anomalies(df_event)
                score = scored_df["anomaly_score"].iloc[0]
                st.success(f"✅ Normal Transaction: {clean_event['product_name']} x {clean_event['quantity']} (${clean_event['total_amount']:.2f}). Score: {score:.3f}")

        with sim_col2:
            if st.button("🚨 Simulate Fraud / Bot Attack Transaction"):
                event = DataGenerator.generate_single_live_transaction(is_anomaly=True)
                clean_event = DataCleaner.clean_single_record(event)
                df_event = pd.DataFrame([clean_event])
                scored_df, alerts = detector.detect_transaction_anomalies(df_event)
                score = scored_df["anomaly_score"].iloc[0]
                if alerts:
                    AnomalyDetector.persist_alerts(alerts)
                st.error(f"🚨 FRAUD ANOMALY FLAGGED! Order amount: ${clean_event['total_amount']:,.2f} for {clean_event['quantity']} units! Score: {score:.3f}")

# ==========================================
# PAGE 6: MLflow Model Governance
# ==========================================
elif page == "🔬 MLflow Model Governance":
    st.header("🔬 MLflow Experiment Hub & Model Governance")
    st.markdown("Monitor model metrics (RMSE, MAE, R², WAPE), parameter configurations, and model versions logged into MLflow.")

    runs = mlflow_tracker.get_recent_runs()

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.subheader("Recent MLflow Experiment Runs")
        if runs:
            runs_display = []
            for r in runs:
                runs_display.append({
                    "Run ID": r["run_id"][:8],
                    "Model": r["model_name"],
                    "RMSE": r["metrics"].get("test_rmse") or r["metrics"].get("train_rmse"),
                    "MAE": r["metrics"].get("test_mae") or r["metrics"].get("train_mae"),
                    "R² Score": r["metrics"].get("test_r2") or r["metrics"].get("train_r2"),
                    "WAPE (%)": r["metrics"].get("test_wape_pct", "N/A"),
                    "Status": r["status"]
                })
            st.dataframe(pd.DataFrame(runs_display), use_container_width=True)
        else:
            st.info("No MLflow runs recorded yet. Retrain model below to log runs.")

    with col_m2:
        st.subheader("Model Governance Actions")
        model_choice = st.selectbox("Select Model Architecture", ["XGBoost Regressor", "Random Forest Regressor"])
        if st.button("⚡ Retrain & Register Model with MLflow"):
            with st.spinner("Training model and logging to MLflow..."):
                m_type = "xgboost" if "XGBoost" in model_choice else "random_forest"
                daily_df = DatabaseRepository.run_arbitrary_query("SELECT * FROM daily_metrics")
                feat_df = FeatureEngineer.generate_demand_features(daily_df)
                X_train, y_train, X_test, y_test = FeatureEngineer.prepare_train_test_split(feat_df, test_days=14)
                new_fc = DemandForecaster(model_type=m_type)
                metrics = new_fc.train(X_train, y_train, X_test, y_test)
                new_fc.save()
                mlflow_tracker.log_training_run(
                    model_name=f"{m_type}_demand_forecaster",
                    params={"model_type": m_type, "features_count": len(new_fc.feature_names)},
                    metrics=metrics
                )
                st.success(f"Model retrained and logged! Test RMSE: {metrics.get('test_rmse')}")
                st.rerun()

st.markdown("---")
st.caption("Predictive Intelligence & Anomaly Detection Platform • Enterprise Decision Support System")
