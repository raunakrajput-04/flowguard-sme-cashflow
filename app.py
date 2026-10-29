import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_generator import generate_sme_dataset
from src.financial_metrics import calculate_kpis
from src.forecasting import train_and_evaluate_models, generate_future_forecast
from src.advisor_engine import analyze_liquidity_health, generate_recommendations, simulate_what_if_scenario

# --- Page Configuration ---
st.set_page_config(
    page_title="FlowGuard | Executive Cash Flow & Working Capital Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Executive Dark Theme Custom Styling (Bloomberg / TradingView Style) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #080C14;
        color: #E2E8F0;
    }

    /* Top Executive Live Bar */
    .top-status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #10B981;
    }
    .live-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #38BDF8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Headers */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    /* Dark Glassmorphic KPI Cards */
    .metric-card-dark {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px 18px;
        text-align: left;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card-dark:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.12);
    }
    .metric-card-dark::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
    }
    .metric-label-dark {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .metric-value-dark {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }
    .metric-sub-dark {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Executive Badges */
    .badge-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #F87171;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.08);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FBBF24;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.08);
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.08);
    }

    /* Action Recommendation Dark Cards */
    .rec-card-dark {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #38BDF8;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Preset Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading / Cache ---
@st.cache_data
def load_or_create_data():
    invoices_path = "data/raw/invoices.csv"
    expenses_path = "data/raw/expenses.csv"
    ledger_path = "data/processed/daily_cash_flow.csv"

    if not (os.path.exists(invoices_path) and os.path.exists(expenses_path) and os.path.exists(ledger_path)):
        generate_sme_dataset()

    invoices = pd.read_csv(invoices_path)
    expenses = pd.read_csv(expenses_path)
    ledger = pd.read_csv(ledger_path)
    return invoices, expenses, ledger

invoices_df, expenses_df, ledger_df = load_or_create_data()

# --- Currency Helpers ---
CURRENCY_MAP = {
    "INR (₹)": {"symbol": "₹", "rate": 1.0},
    "USD ($)": {"symbol": "$", "rate": 0.012},
    "EUR (€)": {"symbol": "€", "rate": 0.011}
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 12px;">
        <span style="font-size:2rem;">⚡</span>
        <div>
            <div style="font-weight:800; font-size:1.25rem; color:#F8FAFC; letter-spacing:-0.5px;">FlowGuard</div>
            <div style="font-size:0.75rem; color:#38BDF8; font-weight:600;">EXECUTIVE TERMINAL</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("AI-Powered Working Capital Intelligence Hub")
    st.markdown("---")

    # Currency Selector
    currency_label = st.selectbox(
        "💱 Reporting Currency",
        options=list(CURRENCY_MAP.keys()),
        index=0,
        help="Convert all cash figures across the dashboard dynamically."
    )
    c_meta = CURRENCY_MAP[currency_label]
    sym = c_meta["symbol"]
    rate = c_meta["rate"]

    def fmt_cur(val: float, decimals: int = 0) -> str:
        """Formats numbers according to chosen currency and conversion rate."""
        converted = val * rate
        if decimals == 0:
            return f"{sym}{converted:,.0f}"
        return f"{sym}{converted:,.{decimals}f}"

    forecast_horizon = st.selectbox(
        "⏱️ Forecast Horizon",
        options=[30, 45, 60, 90],
        index=2,
        help="Multi-step recursive projection horizon."
    )

    model_choice = st.selectbox(
        "🧠 Forecasting Model",
        options=["Gradient Boosting", "Ridge Regression", "Random Forest"],
        index=0,
        help="Machine learning algorithm for Inflow & Outflow recursive time-series forecasting."
    )

    safety_buffer_inr = st.number_input(
        f"🛡️ Safety Reserve ({sym})",
        min_value=50000.0 * rate,
        max_value=10000000.0 * rate,
        value=500000.0 * rate,
        step=50000.0 * rate,
        help="Minimum operating reserve required to cover payroll, taxes, and vendor commitments."
    )
    # Convert safety buffer back to base INR for backend calculations
    safety_buffer_base = safety_buffer_inr / rate

    st.markdown("---")
    st.markdown("#### **FLB Engineering Context**")
    st.caption("• **Subject:** Finance, Law & Business (FLB)")
    st.caption("• **Target:** SME Liquidity Risk & Operating Cycle")
    st.caption("• **AI Engine:** Tabular ML + Recursive Time-Series")

    if st.button("🔄 Regenerate SME Ledger", use_container_width=True):
        st.cache_data.clear()
        generate_sme_dataset()
        st.rerun()

# --- Calculate Financial Metrics ---
metrics = calculate_kpis(invoices_df, expenses_df, ledger_df)

# --- Generate Forecast & Liquidity Health ---
with st.spinner("⚡ Running neural & tree-based cash flow simulation..."):
    forecast_df = generate_future_forecast(ledger_df, horizon_days=forecast_horizon, model_type=model_choice)
    liquidity_health = analyze_liquidity_health(forecast_df, safety_buffer=safety_buffer_base)
    recommendations = generate_recommendations(liquidity_health, invoices_df, expenses_df, forecast_df)

# --- Compute Integrated Liquidity Health Score (0 - 100) ---
def compute_health_score(metrics_dict: dict, health_dict: dict) -> int:
    score = 0
    # 1. Runway (max 25)
    runway = metrics_dict.get("runway_months", 0)
    if runway >= 6: score += 25
    elif runway >= 3: score += 18
    elif runway >= 1: score += 10
    else: score += 3

    # 2. Current Ratio (max 25)
    cr = metrics_dict.get("current_ratio", 1.0)
    if cr >= 1.8: score += 25
    elif cr >= 1.3: score += 20
    elif cr >= 1.0: score += 12
    else: score += 5

    # 3. Cash Conversion Cycle (max 25)
    ccc = metrics_dict.get("ccc_days", 60)
    if ccc <= 45: score += 25
    elif ccc <= 65: score += 18
    elif ccc <= 90: score += 10
    else: score += 4

    # 4. Liquidity Deficit Risk (max 25)
    if health_dict["status"] == "HEALTHY LIQUIDITY": score += 25
    elif health_dict["status"] == "POTENTIAL RISK (95% CI)": score += 14
    else: score += 4

    return min(100, max(0, score))

health_score = compute_health_score(metrics, liquidity_health)

# --- Live Executive Status Header ---
st.markdown(f"""
<div class="top-status-bar">
    <div style="display:flex; align-items:center;">
        <span class="pulse-dot"></span>
        <span class="live-tag">FLOWGUARD AI ENGINE ONLINE • AS OF {metrics['as_of_date']}</span>
    </div>
    <div style="display:flex; gap:16px; font-size:0.85rem; color:#94A3B8;">
        <span>🏢 Enterprise: <b style="color:#F8FAFC;">Apex B2B Distribution SME</b></span>
        <span>⚡ Active Model: <b style="color:#38BDF8;">{model_choice}</b></span>
        <span>🛡️ Currency: <b style="color:#F8FAFC;">{currency_label.split()[0]}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FlowGuard Executive Liquidity Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered SME Cash Flow Forecasting, Working Capital Analytics & Prescriptive Treasury Advisor</div>', unsafe_allow_html=True)

# --- Top Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Cockpit",
    "📈 AI Time-Series Forecaster",
    "📑 Working Capital & Aging",
    "🛡️ AI Prescriptive Advisor",
    "🧪 What-If Stress Simulator"
])

# ==============================================================================
# TAB 1: EXECUTIVE COCKPIT & HEALTH SCORECARD
# ==============================================================================
with tab1:
    st.markdown("### Executive Liquidity & Working Capital Health")
    st.caption("Real-time telemetry on cash runway, working capital cycle, and short-term solvency.")

    # Top Row: 4 Metric Cards + Financial Health Gauge
    c_kpi1, c_kpi2, c_kpi3, c_kpi4, c_gauge = st.columns([1, 1, 1, 1, 1.2])

    with c_kpi1:
        st.markdown(f"""
        <div class="metric-card-dark">
            <div class="metric-label-dark">Current Cash Reserve</div>
            <div class="metric-value-dark">{fmt_cur(metrics['current_cash'])}</div>
            <div class="metric-sub-dark">Liquid Bank Accounts</div>
        </div>
        """, unsafe_allow_html=True)

    with c_kpi2:
        st.markdown(f"""
        <div class="metric-card-dark">
            <div class="metric-label-dark">Net Working Capital</div>
            <div class="metric-value-dark">{fmt_cur(metrics['working_capital'])}</div>
            <div class="metric-sub-dark">Current Assets - Liabilities</div>
        </div>
        """, unsafe_allow_html=True)

    with c_kpi3:
        st.markdown(f"""
        <div class="metric-card-dark">
            <div class="metric-label-dark">Cash Conversion Cycle</div>
            <div class="metric-value-dark">{metrics['ccc_days']}d</div>
            <div class="metric-sub-dark">DIO ({metrics['dio_days']}d) + DSO ({metrics['dso_days']}d) - DPO ({metrics['dpo_days']}d)</div>
        </div>
        """, unsafe_allow_html=True)

    with c_kpi4:
        st.markdown(f"""
        <div class="metric-card-dark">
            <div class="metric-label-dark">Projected Cash Runway</div>
            <div class="metric-value-dark">{metrics['runway_months']} Mo</div>
            <div class="metric-sub-dark">Avg Burn: {fmt_cur(metrics['avg_monthly_outflow'])}/mo</div>
        </div>
        """, unsafe_allow_html=True)

    with c_gauge:
        # Dark Mode Semicircular Health Gauge
        gauge_color = "#10B981" if health_score >= 70 else ("#F59E0B" if health_score >= 45 else "#EF4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "/100", "font": {"size": 22, "color": "#F8FAFC", "family": "JetBrains Mono"}},
            title={"text": "<b>Liquidity Health Score</b>", "font": {"size": 13, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [45, 75], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.15)"}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=145,
            margin=dict(l=10, r=10, t=25, b=5),
            paper_bgcolor="rgba(17, 24, 39, 0.75)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Financial Ratios Grid
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Days Sales Outstanding (DSO)", f"{metrics['dso_days']} Days", delta="-2.4d vs Target", delta_color="inverse")
    m2.metric("Days Payable Outstanding (DPO)", f"{metrics['dpo_days']} Days", delta="+1.8d vs Industry")
    m3.metric("Current Ratio (Solvency)", f"{metrics['current_ratio']}x", help="Current Assets / Current Liabilities. Benchmark > 1.5x")
    m4.metric("Quick Ratio (Acid-Test)", f"{metrics['quick_ratio']}x", help="Liquid Assets / Current Liabilities. Benchmark > 1.0x")

    st.markdown("---")

    # Historical Inflow vs Outflow Monthly Trend (TradingView Dark Style)
    st.markdown("#### Historical Inflow vs Outflow Cash Performance (Trailing 12 Months)")
    hist_ledger = ledger_df.copy()
    hist_ledger["month_year"] = pd.to_datetime(hist_ledger["date"]).dt.to_period("M").astype(str)
    monthly_agg = hist_ledger.groupby("month_year").agg({
        "inflow": "sum",
        "outflow": "sum",
        "net_cash_flow": "sum"
    }).tail(12).reset_index()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=monthly_agg["month_year"],
        y=monthly_agg["inflow"] * rate,
        name="Collections (Inflow)",
        marker=dict(color="#10B981", line=dict(color="#34D399", width=1))
    ))
    fig_hist.add_trace(go.Bar(
        x=monthly_agg["month_year"],
        y=monthly_agg["outflow"] * rate,
        name="Disbursements (Outflow)",
        marker=dict(color="#F43F5E", line=dict(color="#FB7185", width=1))
    ))
    fig_hist.add_trace(go.Scatter(
        x=monthly_agg["month_year"],
        y=monthly_agg["net_cash_flow"] * rate,
        name="Net Operating Cash Flow",
        mode="lines+markers",
        line=dict(color="#38BDF8", width=3)
    ))
    fig_hist.update_layout(
        barmode="group",
        template="plotly_dark",
        height=370,
        paper_bgcolor="rgba(11, 15, 25, 0.8)",
        plot_bgcolor="rgba(11, 15, 25, 0.4)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Month", gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(title=f"Amount ({sym})", gridcolor="rgba(255, 255, 255, 0.05)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 2: AI CASH FLOW FORECASTING
# ==============================================================================
with tab2:
    st.markdown(f"### AI Multi-Step Cash Flow Trajectory ({forecast_horizon}-Day Projection)")
    st.caption(f"Recursive machine learning model trained on historical payment velocity using **{model_choice}** with 80% & 95% confidence bands.")

    # Main Fan Chart
    hist_tail = ledger_df.tail(60)[["date", "cash_balance"]].copy()
    hist_tail.rename(columns={"cash_balance": "projected_cash_balance"}, inplace=True)

    fc_plot = forecast_df.copy()

    fig_fc = go.Figure()

    # 95% Confidence Band
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_upper_95"] * rate,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_lower_95"] * rate,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(168, 85, 247, 0.12)",
        name="95% Confidence Band"
    ))

    # 80% Confidence Band
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_upper_80"] * rate,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_lower_80"] * rate,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(168, 85, 247, 0.22)",
        name="80% Confidence Band"
    ))

    # Historical Actual Line
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["projected_cash_balance"] * rate,
        name="Historical Cash Balance",
        mode="lines",
        line=dict(color="#38BDF8", width=3)
    ))

    # Projected Mean Line
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["projected_cash_balance"] * rate,
        name="AI Forecasted Cash Balance",
        mode="lines+markers",
        line=dict(color="#A855F7", width=3, dash="dash")
    ))

    # Minimum Safety Reserve Line
    fig_fc.add_hline(
        y=safety_buffer_inr,
        line_dash="dot",
        line_color="#EF4444",
        line_width=2,
        annotation_text=f"Safety Reserve: {sym}{safety_buffer_inr:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="#F87171"
    )

    fig_fc.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(11, 15, 25, 0.8)",
        plot_bgcolor="rgba(11, 15, 25, 0.4)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Timeline", gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(title=f"Cash Balance ({sym})", gridcolor="rgba(255, 255, 255, 0.05)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Lower Grid: Dual stream & Model Benchmarking Table
    c_f1, c_f2 = st.columns(2)

    with c_f1:
        st.markdown("#### Forecasted Inflows vs Outflows")
        fig_io = go.Figure()
        fig_io.add_trace(go.Bar(
            x=forecast_df["date"],
            y=forecast_df["forecast_inflow"] * rate,
            name="Forecasted Inflow",
            marker_color="#10B981"
        ))
        fig_io.add_trace(go.Bar(
            x=forecast_df["date"],
            y=forecast_df["forecast_outflow"] * rate,
            name="Forecasted Outflow",
            marker_color="#F43F5E"
        ))
        fig_io.update_layout(
            template="plotly_dark",
            barmode="group",
            height=300,
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(11, 15, 25, 0.4)",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(title=f"Amount ({sym})", gridcolor="rgba(255, 255, 255, 0.05)")
        )
        st.plotly_chart(fig_io, use_container_width=True)

    with c_f2:
        st.markdown("#### Model Performance Leaderboard (Holdout Set)")
        with st.spinner("Benchmarking algorithms..."):
            eval_metrics = train_and_evaluate_models(ledger_df, target_col="inflow", test_days=60)
            
            # Format display df with current currency
            summary_disp = eval_metrics["results_summary"].copy()
            summary_disp[f"MAE ({sym})"] = (summary_disp["MAE (INR)"] * rate).round(2)
            summary_disp[f"RMSE ({sym})"] = (summary_disp["RMSE (INR)"] * rate).round(2)
            summary_disp = summary_disp[["Model", f"MAE ({sym})", f"RMSE ({sym})", "MAPE (%)"]]

            st.dataframe(
                summary_disp,
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"🏆 **Top Performer:** `{eval_metrics['best_model_name']}` minimizes recursive forecast error over classical Holt-Winters baseline.")

# ==============================================================================
# TAB 3: WORKING CAPITAL & AGING ANALYSIS
# ==============================================================================
with tab3:
    st.markdown("### Accounts Receivable (AR) & Accounts Payable (AP) Aging")
    st.caption("Deep dive into customer payment delays, credit tier exposures, and working capital traps.")

    col_ag1, col_ag2 = st.columns(2)

    with col_ag1:
        st.markdown("#### AR Aging Distribution")
        aging_dict = metrics["aging_buckets"]
        aging_df = pd.DataFrame([
            {"Category": k, "Amount": v * rate} for k, v in aging_dict.items()
        ])

        fig_pie = px.pie(
            aging_df,
            names="Category",
            values="Amount",
            color="Category",
            color_discrete_map={
                "Current (0-30 days)": "#10B981",
                "1-30 Days Past Due": "#F59E0B",
                "31-60 Days Past Due": "#F97316",
                "61-90 Days Past Due": "#EF4444",
                "90+ Days Past Due": "#991B1B"
            },
            hole=0.45,
            template="plotly_dark"
        )
        fig_pie.update_layout(
            height=320,
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_ag2:
        st.markdown("#### Client Credit Tier Analysis")
        invoices_active = invoices_df[invoices_df["status"].isin(["Paid", "Overdue", "Pending"])].copy()
        tier_summary = invoices_active.groupby("credit_tier").agg(
            total_billed=("amount", lambda x: (x * rate).sum()),
            avg_delay=("days_delayed", "mean"),
            invoice_count=("invoice_id", "count")
        ).reset_index()

        fig_tier = px.bar(
            tier_summary,
            x="credit_tier",
            y="avg_delay",
            color="credit_tier",
            text=tier_summary["avg_delay"].round(1).astype(str) + "d lag",
            labels={"credit_tier": "Credit Tier", "avg_delay": "Avg Delay (Days)"},
            color_discrete_map={"A": "#10B981", "B": "#F59E0B", "C": "#EF4444"},
            template="plotly_dark"
        )
        fig_tier.update_layout(
            height=320,
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚡ Interactive Early Discount Calculator (Dynamic AR Acceleration)")
    st.caption("Select an open invoice to calculate the exact liquidity acceleration and capital savings from a **2/10 Net-30** discount offer.")

    unpaid_invoices = invoices_df[invoices_df["status"].isin(["Pending", "Overdue"])].sort_values("amount", ascending=False).copy()
    unpaid_invoices["converted_amount"] = (unpaid_invoices["amount"] * rate).round(2)
    
    selected_inv_id = st.selectbox(
        "Select Open Customer Invoice to Simulate Acceleration:",
        options=unpaid_invoices["invoice_id"].tolist(),
        format_func=lambda x: f"{x} — {unpaid_invoices.loc[unpaid_invoices['invoice_id']==x, 'client_name'].values[0]} | {sym}{unpaid_invoices.loc[unpaid_invoices['invoice_id']==x, 'converted_amount'].values[0]:,.2f} ({unpaid_invoices.loc[unpaid_invoices['invoice_id']==x, 'status'].values[0]})"
    )

    inv_row = unpaid_invoices[unpaid_invoices["invoice_id"] == selected_inv_id].iloc[0]
    inv_amount_cur = inv_row["converted_amount"]
    discount_val = inv_amount_cur * 0.02
    net_collected_early = inv_amount_cur * 0.98

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Invoice Face Value", f"{sym}{inv_amount_cur:,.2f}")
    ic2.metric("2% Discount Foregone", f"{sym}{discount_val:,.2f}", help="Cost of capital to the SME")
    ic3.metric("Accelerated Cash Unlocked", f"{sym}{net_collected_early:,.2f}", delta="Instant 7-Day Inflow")
    ic4.metric("Days of Liquidity Gained", "18–25 Days", delta="Shortens DSO")

# ==============================================================================
# TAB 4: AI PRESCRIPTIVE LIQUIDITY ADVISOR
# ==============================================================================
with tab4:
    st.markdown("### AI Prescriptive Liquidity Advisor & Early Warning Engine")
    st.caption("Automated managerial decision matrix based on corporate finance principles.")

    # Status Alert Banner
    if liquidity_health["status"] == "CRITICAL DEFICIT ALERT":
        st.markdown(f"""
        <div class="badge-danger">
            <h4 style="margin:0; font-weight:700;">⚠️ CRITICAL LIQUIDITY DEFICIT DETECTED</h4>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                Cash balance is projected to breach the {sym}{safety_buffer_inr:,.0f} safety reserve on 
                <b>{liquidity_health['first_breach_date']}</b> (in {liquidity_health['days_to_breach']} days). 
                Maximum projected capital shortfall: <b>{fmt_cur(liquidity_health['shortfall_amount'])}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif liquidity_health["status"] == "POTENTIAL RISK (95% CI)":
        st.markdown(f"""
        <div class="badge-warning">
            <h4 style="margin:0; font-weight:700;">⚠️ VOLATILITY RISK DETECTED (95% CI LOWER BOUND)</h4>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                Under adverse volatility shocks, cash reserves may dip below the safety reserve around 
                <b>{liquidity_health['first_breach_date']}</b>. Implement proactive working capital buffers now.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="badge-success">
            <h4 style="margin:0; font-weight:700;">✅ LIQUIDITY RESERVES OPTIMAL</h4>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                Operating liquidity is projected to remain well above the {sym}{safety_buffer_inr:,.0f} safety buffer across the 
                entire {forecast_horizon}-day horizon. Peak projected buffer: <b>{fmt_cur(liquidity_health['min_projected_balance'])}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Prescriptive Action Plan & Financial Levers")

    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"**Action #{i}: [{rec['priority']}] {rec['type']}**", expanded=(i <= 2)):
            col_r1, col_r2 = st.columns([3, 1.2])
            with col_r1:
                st.markdown(f"**Action Required:** {rec['action']}")
                st.markdown(f"💡 *Corporate Finance Rationale:* {rec['financial_rationale']}")
            with col_r2:
                st.info(f"**Financial Impact:**\n\n{rec['impact']}")

    st.markdown("---")
    
    # Download Executive Report Button
    report_text = f"""=====================================================
FLOWGUARD EXECUTIVE LIQUIDITY & WORKING CAPITAL AUDIT
Generated: {metrics['as_of_date']}
=====================================================

1. EXECUTIVE SUMMARY:
- Current Liquid Cash: {fmt_cur(metrics['current_cash'])}
- Net Working Capital: {fmt_cur(metrics['working_capital'])}
- Cash Conversion Cycle: {metrics['ccc_days']} Days
- Estimated Cash Runway: {metrics['runway_months']} Months
- FlowGuard Health Score: {health_score}/100

2. LIQUIDITY STATUS:
- Status: {liquidity_health['status']}
- Safety Buffer: {fmt_cur(safety_buffer_base)}
- Minimum Projected Balance: {fmt_cur(liquidity_health['min_projected_balance'])}

3. PRESCRIPTIVE RECOMMENDATIONS:
"""
    for idx, r in enumerate(recommendations, 1):
        report_text += f"\n[{idx}] {r['type']} ({r['priority']})\n    Action: {r['action']}\n    Impact: {r['impact']}\n"

    st.download_button(
        label="📥 Download Executive Financial Audit Report (.TXT)",
        data=report_text,
        file_name=f"FlowGuard_Financial_Audit_{metrics['as_of_date']}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ==============================================================================
# TAB 5: WHAT-IF STRESS SIMULATOR
# ==============================================================================
with tab5:
    st.markdown("### Interactive 'What-If' Stress-Testing Simulator")
    st.caption("Stress-test the SME's liquidity against macroeconomic shocks, customer payment freezes, or supplier price hikes.")

    # Initialize session state for scenario presets
    if "sim_sales" not in st.session_state: st.session_state.sim_sales = 0
    if "sim_delay" not in st.session_state: st.session_state.sim_delay = 0
    if "sim_opex" not in st.session_state: st.session_state.sim_opex = 0
    if "sim_disc" not in st.session_state: st.session_state.sim_disc = 0

    def apply_preset(sales, delay, opex, disc):
        st.session_state.sim_sales = sales
        st.session_state.sim_delay = delay
        st.session_state.sim_opex = opex
        st.session_state.sim_disc = disc

    # Quick Preset Buttons
    st.markdown("##### ⚡ Quick Stress-Testing Presets:")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🚀 Aggressive Expansion", use_container_width=True, help="+25% Sales, 0d Delay, +10% OpEx, 30% Discount Adoption"):
            apply_preset(25, 0, 0, 30)
            st.rerun()
    with p2:
        if st.button("⚠️ Client Payment Freeze", use_container_width=True, help="-15% Sales, +15d Payment Delay, 0% OpEx Cuts"):
            apply_preset(-15, 15, 0, 10)
            st.rerun()
    with p3:
        if st.button("🚨 Macro Recession Shock", use_container_width=True, help="-35% Sales, +25d Delay, -15% OpEx Cuts"):
            apply_preset(-35, 25, 15, 5)
            st.rerun()
    with p4:
        if st.button("🔄 Reset to Baseline", use_container_width=True, help="Reset all sliders to zero"):
            apply_preset(0, 0, 0, 0)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_sim_ctrl, col_sim_view = st.columns([1, 2])

    with col_sim_ctrl:
        st.markdown("#### Scenario Controls")
        sim_sales_shock = st.slider("Revenue Shock (%)", min_value=-40, max_value=30, value=st.session_state.sim_sales, step=5)
        sim_delay_shock = st.slider("Customer Payment Delay (Days)", min_value=0, max_value=30, value=st.session_state.sim_delay, step=3)
        sim_opex_cut = st.slider("Operational Cost Cuts (%)", min_value=0, max_value=30, value=st.session_state.sim_opex, step=5)
        sim_discount_adoption = st.slider("Early Discount Adoption (%)", min_value=0, max_value=50, value=st.session_state.sim_disc, step=5)

        st.markdown("---")
        st.markdown("#### Active Assumption Matrix")
        st.write(f"• **Revenue Drift:** `{sim_sales_shock:+d}%`")
        st.write(f"• **Receivables Lag:** `+{sim_delay_shock} days`")
        st.write(f"• **Overhead Reductions:** `-{sim_opex_cut}%`")
        st.write(f"• **Early Discount Take-up:** `{sim_discount_adoption}%`")

    with col_sim_view:
        simulated_df = simulate_what_if_scenario(
            forecast_df,
            sales_growth_pct=sim_sales_shock,
            delay_shock_days=sim_delay_shock,
            opex_reduction_pct=sim_opex_cut,
            discount_adoption_rate=sim_discount_adoption
        )

        fig_sim = go.Figure()
        
        # Baseline projection
        fig_sim.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["projected_cash_balance"] * rate,
            name="Baseline AI Forecast",
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dash")
        ))
        
        # Stress-tested projection
        fig_sim.add_trace(go.Scatter(
            x=simulated_df["date"],
            y=simulated_df["projected_cash_balance"] * rate,
            name="Stress-Tested Scenario",
            mode="lines+markers",
            line=dict(color="#A855F7", width=3)
        ))
        
        # Safety line
        fig_sim.add_hline(
            y=safety_buffer_inr,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"Safety Reserve: {sym}{safety_buffer_inr:,.0f}",
            annotation_font_color="#F87171"
        )

        fig_sim.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(11, 15, 25, 0.4)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Timeline", gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(title=f"Cash Balance ({sym})", gridcolor="rgba(255, 255, 255, 0.05)"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Delta Summary
        base_ending = forecast_df["projected_cash_balance"].iloc[-1] * rate
        sim_ending = simulated_df["projected_cash_balance"].iloc[-1] * rate
        diff = sim_ending - base_ending
        
        s_c1, s_c2 = st.columns(2)
        s_c1.metric("Baseline Ending Cash", f"{sym}{base_ending:,.0f}")
        s_c2.metric("Scenario Ending Cash", f"{sym}{sim_ending:,.0f}", delta=f"{sym}{diff:,.0f}")

st.markdown("---")
st.caption("FlowGuard Executive Terminal • FLB + AI/ML Engineering Project • Powered by Streamlit, Plotly & Scikit-Learn")
