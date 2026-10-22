from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import config
from src.logger import logger
from src.database.connection import init_db
from src.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {config.app_name} v{config.version}...")
    init_db()
    yield
    logger.info("Shutting down platform API...")

app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Enterprise ML platform predicting business demand, detecting unusual patterns, and explaining prediction drivers.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Intelligence & Anomaly Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #f1f5f9; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 20px; margin-bottom: 28px; }
        .logo { font-size: 1.35rem; font-weight: 700; color: #6366f1; display: flex; align-items: center; gap: 10px; }
        .links a { color: #94a3b8; text-decoration: none; margin-left: 18px; font-weight: 500; font-size: 0.9rem; transition: color 0.2s; }
        .links a:hover { color: #f8fafc; }
        .btn { background: #6366f1; color: white; padding: 8px 16px; border-radius: 8px; text-decoration: none; font-size: 0.88rem; font-weight: 600; border: none; cursor: pointer; }
        .btn:hover { background: #4f46e5; }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin-bottom: 28px; }
        .card { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; backdrop-filter: blur(10px); }
        .card .title { font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 6px; }
        .card .val { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
        .card .sub { font-size: 0.8rem; margin-top: 4px; }
        .text-green { color: #10b981; }
        .text-red { color: #ef4444; }
        .panel { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
        .panel h3 { margin-bottom: 16px; font-size: 1.1rem; color: #f8fafc; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { text-align: left; padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.88rem; }
        th { color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
        .badge-CRITICAL { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
        .badge-HIGH { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
        .badge-MEDIUM { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
        .flex { display: flex; gap: 12px; align-items: center; }
        input, select { background: #0f172a; border: 1px solid rgba(255,255,255,0.15); color: #f8fafc; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; }
        .result-box { margin-top: 16px; padding: 14px; background: rgba(15,23,42,0.8); border-radius: 8px; border-left: 3px solid #6366f1; font-size: 0.92rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">⚡ Predictive Intelligence & Anomaly Platform</div>
            <div class="links">
                <a href="/docs" target="_blank">Swagger API Docs</a>
                <a href="/api/health" target="_blank">Health Check</a>
                <a href="https://github.com/infant88/Predictive-Intelligence-Anomaly-Detection-Platform" target="_blank">GitHub</a>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="grid-4" id="kpis">
            <div class="card"><div class="title">Total Revenue</div><div class="val" id="kpi-rev">$...</div><div class="sub text-green">Live Database</div></div>
            <div class="card"><div class="title">Total Orders</div><div class="val" id="kpi-orders">...</div><div class="sub">Completed Transactions</div></div>
            <div class="card"><div class="title">Active Alerts</div><div class="val text-red" id="kpi-alerts">...</div><div class="sub">Open Anomalies</div></div>
            <div class="card"><div class="title">System Status</div><div class="val text-green">ONLINE</div><div class="sub">FastAPI + XGBoost + IsoForest</div></div>
        </div>

        <!-- Demand Forecasting Panel -->
        <div class="panel">
            <h3>🔮 Demand Forecasting & What-If Promotion Simulator</h3>
            <div class="flex" style="flex-wrap: wrap; margin-bottom: 16px;">
                <select id="prod-select">
                    <option value="PROD-TECH-001">Wireless Headphones (PROD-TECH-001)</option>
                    <option value="PROD-TECH-002">4K Smart Monitor (PROD-TECH-002)</option>
                    <option value="PROD-APPR-001">Wool Sweater (PROD-APPR-001)</option>
                    <option value="PROD-HOME-001">Smart Multi-Cooker (PROD-HOME-001)</option>
                </select>
                <input type="number" id="horizon-input" value="7" min="1" max="30" placeholder="Horizon Days" style="width: 140px;">
                <input type="number" id="promo-input" value="15" min="0" max="50" placeholder="Promo Discount %" style="width: 170px;">
                <button class="btn" onclick="runForecast()">Run Forecast & Explain</button>
            </div>
            <div id="forecast-result" class="result-box" style="display: none;"></div>
        </div>

        <!-- Live Fraud Screener Panel -->
        <div class="panel">
            <h3>💳 Live Streaming Transaction Fraud Screener (Isolation Forest)</h3>
            <div class="flex" style="flex-wrap: wrap;">
                <input type="number" id="stream-qty" value="3" placeholder="Quantity" style="width: 120px;">
                <input type="number" id="stream-price" value="149.99" placeholder="Unit Price ($)" style="width: 140px;">
                <button class="btn" onclick="streamEvent(false)">⚡ Screen Normal Order</button>
                <button class="btn" style="background: #ef4444;" onclick="streamEvent(true)">🚨 Screen Fraud / Bot Attack</button>
            </div>
            <div id="stream-result" class="result-box" style="display: none;"></div>
        </div>

        <!-- Anomaly Alerts Table -->
        <div class="panel">
            <h3>🚨 Recent Anomaly Alerts Feed</h3>
            <table>
                <thead><tr><th>Time</th><th>Type</th><th>Entity</th><th>Severity</th><th>Description</th><th>Action</th></tr></thead>
                <tbody id="alerts-body"><tr><td colspan="6">Loading alerts...</td></tr></tbody>
            </table>
        </div>
    </div>

    <script>
        async function loadKPIs() {
            try {
                const res = await fetch('/api/kpis');
                const data = await res.json();
                document.getElementById('kpi-rev').innerText = '$' + Number(data.total_revenue).toLocaleString();
                document.getElementById('kpi-orders').innerText = Number(data.total_orders).toLocaleString();
                document.getElementById('kpi-alerts').innerText = data.open_alerts;
            } catch(e) { console.error(e); }
        }

        async function loadAlerts() {
            try {
                const res = await fetch('/api/anomalies/alerts?limit=6');
                const alerts = await res.json();
                const tbody = document.getElementById('alerts-body');
                if (!alerts || alerts.length === 0) { tbody.innerHTML = '<tr><td colspan="6">No alerts found.</td></tr>'; return; }
                tbody.innerHTML = alerts.map(a => `
                    <tr>
                        <td>${a.detected_at}</td>
                        <td><strong>${a.alert_type}</strong></td>
                        <td>${a.entity_id}</td>
                        <td><span class="badge badge-${a.severity}">${a.severity}</span></td>
                        <td>${a.description}</td>
                        <td style="color: #6366f1;">${a.action_item}</td>
                    </tr>
                `).join('');
            } catch(e) { console.error(e); }
        }

        async function runForecast() {
            const pid = document.getElementById('prod-select').value;
            const horizon = parseInt(document.getElementById('horizon-input').value) || 7;
            const promo = parseFloat(document.getElementById('promo-input').value) || 0;
            const box = document.getElementById('forecast-result');
            box.style.display = 'block';
            box.innerHTML = 'Calculating recursive gradient-boosted demand forecast...';

            try {
                const res = await fetch('/api/forecast/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({product_id: pid, horizon_days: horizon, promo_discount_pct: promo})
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Forecast failed');
                
                const avgPred = (data.points.reduce((s, p) => s + p.predicted_quantity, 0) / data.points.length).toFixed(1);
                const totalPred = data.points.reduce((s, p) => s + p.predicted_quantity, 0).toFixed(0);

                // Fetch XAI explanation
                const expRes = await fetch('/api/forecast/explain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({product_id: pid})
                });
                const expData = await expRes.json();

                box.innerHTML = `
                    <strong>Forecast Results for ${pid}:</strong><br>
                    • Projected Avg Daily Demand: <strong>${avgPred} units/day</strong> (Total ${horizon}-day volume: <strong>${totalPred} units</strong> with ${promo}% promo).<br>
                    • Prediction Confidence: <strong>95%</strong> (Lower bound: ${data.points[0].lower_bound} to Upper bound: ${data.points[0].upper_bound}).<br>
                    <div style="margin-top: 8px; color: #94a3b8;"><strong>🧠 SHAP Factor Attribution:</strong> ${expData.executive_narrative || 'Calculated via TreeExplainer.'}</div>
                `;
            } catch(e) {
                box.innerHTML = '<span style="color: #ef4444;">Error: ' + e.message + '</span>';
            }
        }

        async function streamEvent(isFraud) {
            const box = document.getElementById('stream-result');
            box.style.display = 'block';
            box.innerHTML = 'Screening transaction through Isolation Forest model...';
            const qty = isFraud ? 450 : (parseFloat(document.getElementById('stream-qty').value) || 2);
            const price = parseFloat(document.getElementById('stream-price').value) || 149.99;

            try {
                const res = await fetch('/api/data/stream-event', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        stock_code: 'PROD-TECH-001',
                        quantity: qty,
                        unit_price: price,
                        customer_id: isFraud ? 'CUST-FRAUD-BOT' : 'CUST-NORMAL'
                    })
                });
                const data = await res.json();
                if (data.is_anomaly) {
                    box.innerHTML = `<span style="color: #ef4444; font-weight: 700;">🚨 FRAUD ANOMALY FLAGGED!</span> Anomaly Score: <strong>${data.anomaly_score}</strong>. Total Order: $${data.cleaned_transaction.total_amount}. Action: Fulfillment held for fraud review.`;
                } else {
                    box.innerHTML = `<span style="color: #10b981; font-weight: 700;">✅ Transaction Verified Clean.</span> Anomaly Score: <strong>${data.anomaly_score}</strong> (Normal threshold). Processed successfully.`;
                }
                loadKPIs();
                loadAlerts();
            } catch(e) {
                box.innerHTML = '<span style="color: #ef4444;">Error: ' + e.message + '</span>';
            }
        }

        loadKPIs();
        loadAlerts();
    </script>
</body>
</html>
"""

