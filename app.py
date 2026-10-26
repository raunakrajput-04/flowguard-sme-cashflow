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
    page_title="FlowGuard | SME Cash Flow & Working Capital Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 4px;
    }
    .alert-card-danger {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .alert-card-success {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .alert-card-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
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

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/line-chart.png", width=64)
    st.markdown("### **FlowGuard Controls**")
    st.caption("AI-Powered Working Capital Intelligence Hub")
    
    st.markdown("---")
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        options=[30, 45, 60, 90],
        index=2,
        help="Number of days to forecast cash inflows, outflows, and ending balance."
    )

    model_choice = st.selectbox(
        "Forecasting Model",
        options=["Gradient Boosting", "Ridge Regression", "Random Forest"],
        index=0,
        help="Machine learning algorithm used for recursive multi-step forecasting."
    )

    safety_buffer = st.number_input(
        "Safety Cash Buffer (INR)",
        min_value=100000.0,
        max_value=5000000.0,
        value=500000.0,
        step=50000.0,
        help="Minimum operating reserve required to cover payroll, rent, and unforeseen shocks."
    )

    st.markdown("---")
    st.markdown("#### **FLB Subject Info**")
    st.caption("**Domain:** Corporate Finance & Business Operations")
    st.caption("**Target:** SME Liquidity Management & Working Capital")
    st.caption("**AI Focus:** Time-Series Forecasting & Prescriptive XAI")
    
    if st.button("🔄 Regenerate Sample Data"):
        st.cache_data.clear()
        generate_sme_dataset()
        st.rerun()

# --- Calculate Financial Metrics ---
metrics = calculate_kpis(invoices_df, expenses_df, ledger_df)

# --- Generate Forecast & Liquidity Health ---
with st.spinner("Computing time-series cash flow projections..."):
    forecast_df = generate_future_forecast(ledger_df, horizon_days=forecast_horizon, model_type=model_choice)
    liquidity_health = analyze_liquidity_health(forecast_df, safety_buffer=safety_buffer)
    recommendations = generate_recommendations(liquidity_health, invoices_df, expenses_df, forecast_df)

# --- Main App Header ---
st.markdown('<div class="main-header">FlowGuard: SME Cash Flow & Working Capital Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Liquidity Forecasting, Working Capital Optimization & Prescriptive Treasury Advisor</div>', unsafe_allow_html=True)

# --- Top Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPI Dashboard",
    "📈 AI Cash Flow Forecasting",
    "📑 Working Capital & Aging",
    "🛡️ AI Liquidity Advisor",
    "🧪 What-If Stress Simulator"
])

# ==============================================================================
# TAB 1: EXECUTIVE KPI DASHBOARD
# ==============================================================================
with tab1:
    st.subheader("Executive Financial Health & Working Capital Metrics")
    st.caption("Real-time assessment of liquidity, operational cycle, and cash runway.")

    # 4 Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Cash Balance</div>
            <div class="metric-value">₹{metrics['current_cash']:,.0f}</div>
            <div class="metric-sub">Liquid funds in operating account</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Working Capital</div>
            <div class="metric-value">₹{metrics['working_capital']:,.0f}</div>
            <div class="metric-sub">Current Assets - Current Liabilities</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Conversion Cycle (CCC)</div>
            <div class="metric-value">{metrics['ccc_days']} Days</div>
            <div class="metric-sub">DIO ({metrics['dio_days']}d) + DSO ({metrics['dso_days']}d) - DPO ({metrics['dpo_days']}d)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Cash Runway</div>
            <div class="metric-value">{metrics['runway_months']} Months</div>
            <div class="metric-sub">At avg monthly burn: ₹{metrics['avg_monthly_outflow']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Days Sales Outstanding (DSO)", f"{metrics['dso_days']} Days", delta="-2.4 d vs benchmark", delta_color="inverse")
    m_col2.metric("Days Payable Outstanding (DPO)", f"{metrics['dpo_days']} Days", delta="+1.8 d vs benchmark")
    m_col3.metric("Current Ratio", f"{metrics['current_ratio']}x", help="Standard benchmark is > 1.5x")
    m_col4.metric("Quick Ratio (Acid-Test)", f"{metrics['quick_ratio']}x", help="Standard benchmark is > 1.0x")

    st.markdown("---")

    # Historical Inflow vs Outflow Monthly Trend
    st.markdown("### Historical Cash Flow Performance (Last 12 Months)")
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
        y=monthly_agg["inflow"],
        name="Collections (Inflows)",
        marker_color="#10B981"
    ))
    fig_hist.add_trace(go.Bar(
        x=monthly_agg["month_year"],
        y=monthly_agg["outflow"],
        name="Disbursements (Outflows)",
        marker_color="#EF4444"
    ))
    fig_hist.add_trace(go.Scatter(
        x=monthly_agg["month_year"],
        y=monthly_agg["net_cash_flow"],
        name="Net Cash Flow",
        mode="lines+markers",
        line=dict(color="#3B82F6", width=3)
    ))
    fig_hist.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Month",
        yaxis_title="Amount (INR)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 2: AI CASH FLOW FORECASTING
# ==============================================================================
with tab2:
    st.subheader(f"AI Cash Flow Trajectory ({forecast_horizon}-Day Projection)")
    st.caption(f"Recursive multi-step forecasting generated using **{model_choice}** with historical variance confidence bands.")

    # Main Chart: Historical Ending Balance + Projected Balance with Confidence Bands
    hist_tail = ledger_df.tail(60)[["date", "cash_balance"]].copy()
    hist_tail.rename(columns={"cash_balance": "projected_cash_balance"}, inplace=True)
    hist_tail["type"] = "Historical Actual"

    fc_plot = forecast_df.copy()
    fc_plot["type"] = "AI Forecast"

    fig_fc = go.Figure()

    # 95% Confidence Interval Band
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_upper_95"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_lower_95"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.12)",
        name="95% Confidence Band"
    ))

    # 80% Confidence Interval Band
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_upper_80"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["cash_balance_lower_80"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.22)",
        name="80% Confidence Band"
    ))

    # Historical Actual Line
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["projected_cash_balance"],
        name="Historical Cash Balance",
        mode="lines",
        line=dict(color="#0F172A", width=3)
    ))

    # Projected Mean Forecast
    fig_fc.add_trace(go.Scatter(
        x=fc_plot["date"],
        y=fc_plot["projected_cash_balance"],
        name="Projected Cash Balance",
        mode="lines+markers",
        line=dict(color="#2563EB", width=3, dash="dash")
    ))

    # Safety Buffer Threshold Line
    fig_fc.add_hline(
        y=safety_buffer,
        line_dash="dot",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Safety Buffer: ₹{safety_buffer:,.0f}",
        annotation_position="bottom right"
    )

    fig_fc.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timeline",
        yaxis_title="Cash Balance (INR)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Inflow vs Outflow Forecast
    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        st.markdown("#### Projected Daily Inflows & Outflows")
        fig_io = go.Figure()
        fig_io.add_trace(go.Bar(
            x=forecast_df["date"],
            y=forecast_df["forecast_inflow"],
            name="Forecasted Inflows",
            marker_color="#10B981"
        ))
        fig_io.add_trace(go.Bar(
            x=forecast_df["date"],
            y=forecast_df["forecast_outflow"],
            name="Forecasted Outflows",
            marker_color="#F43F5E"
        ))
        fig_io.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_io, use_container_width=True)

    with col_fc2:
        st.markdown("#### Model Performance Benchmarking (Validation Set)")
        with st.spinner("Evaluating models on 60-day holdout set..."):
            eval_metrics = train_and_evaluate_models(ledger_df, target_col="inflow", test_days=60)
            st.dataframe(
                eval_metrics["results_summary"],
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"🏆 **Best Performing Model:** `{eval_metrics['best_model_name']}` based on lowest Mean Absolute Error (MAE).")

# ==============================================================================
# TAB 3: WORKING CAPITAL & AGING ANALYSIS
# ==============================================================================
with tab3:
    st.subheader("Working Capital Breakdown & Aging Ledger")
    st.caption("Deep dive into Accounts Receivable (AR) delays and Accounts Payable (AP) commitments.")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Accounts Receivable Aging Distribution")
        aging_dict = metrics["aging_buckets"]
        aging_df = pd.DataFrame(list(aging_dict.items()), columns=["Aging Category", "Uncollected Amount"])
        
        fig_pie = px.pie(
            aging_df,
            names="Aging Category",
            values="Uncollected Amount",
            color="Aging Category",
            color_discrete_map={
                "Current (0-30 days)": "#10B981",
                "1-30 Days Past Due": "#F59E0B",
                "31-60 Days Past Due": "#F97316",
                "61-90 Days Past Due": "#EF4444",
                "90+ Days Past Due": "#991B1B"
            },
            hole=0.45
        )
        fig_pie.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Client Credit Tier Analysis")
        invoices_active = invoices_df[invoices_df["status"].isin(["Paid", "Overdue", "Pending"])].copy()
        tier_summary = invoices_active.groupby("credit_tier").agg(
            total_billed=("amount", "sum"),
            avg_delay=("days_delayed", "mean"),
            invoice_count=("invoice_id", "count")
        ).reset_index()

        fig_tier = px.bar(
            tier_summary,
            x="credit_tier",
            y="avg_delay",
            color="credit_tier",
            text=tier_summary["avg_delay"].round(1).astype(str) + " days",
            labels={"credit_tier": "Credit Risk Tier", "avg_delay": "Average Days Delayed"},
            color_discrete_map={"A": "#10B981", "B": "#F59E0B", "C": "#EF4444"}
        )
        fig_tier.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("#### Open Invoices Requiring Management Attention")
    unpaid_invoices = invoices_df[invoices_df["status"].isin(["Pending", "Overdue"])].sort_values("amount", ascending=False).head(10)
    st.dataframe(
        unpaid_invoices[["invoice_id", "client_name", "credit_tier", "amount", "due_date", "payment_terms", "status"]],
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# TAB 4: AI LIQUIDITY ADVISOR
# ==============================================================================
with tab4:
    st.subheader("AI Prescriptive Liquidity Advisor & Early Warning Engine")
    st.caption("Automated financial decision recommendations to safeguard working capital.")

    # Status Alert Banner
    if liquidity_health["status"] == "CRITICAL DEFICIT ALERT":
        st.markdown(f"""
        <div class="alert-card-danger">
            <h4 style="margin:0; color:#B91C1C;">⚠️ CRITICAL LIQUIDITY DEFICIT DETECTED</h4>
            <p style="margin:4px 0 0 0; color:#7F1D1D;">
                Cash balance is projected to breach the ₹{safety_buffer:,.0f} safety reserve on 
                <b>{liquidity_health['first_breach_date']}</b> (in {liquidity_health['days_to_breach']} days). 
                Peak projected shortfall: <b>₹{liquidity_health['shortfall_amount']:,.2f}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif liquidity_health["status"] == "POTENTIAL RISK (95% CI)":
        st.markdown(f"""
        <div class="alert-card-warning">
            <h4 style="margin:0; color:#B45309;">⚠️ POTENTIAL VOLATILITY RISK DETECTED</h4>
            <p style="margin:4px 0 0 0; color:#78350F;">
                Under adverse conditions (95% confidence lower bound), cash reserves may drop below the safety buffer 
                around <b>{liquidity_health['first_breach_date']}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card-success">
            <h4 style="margin:0; color:#047857;">✅ LIQUIDITY RESERVE HEALTHY</h4>
            <p style="margin:4px 0 0 0; color:#064E3B;">
                Cash balance is projected to remain well above the ₹{safety_buffer:,.0f} safety buffer across the 
                entire {forecast_horizon}-day horizon. Minimum projected balance: <b>₹{liquidity_health['min_projected_balance']:,.2f}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Prescriptive Recommendations
    st.markdown("### Prescriptive Action Plan")
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"**Action {i}: [{rec['priority']}] {rec['type']}**", expanded=(i <= 2)):
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                st.markdown(f"**Recommended Action:** {rec['action']}")
                st.markdown(f"💡 *Financial Rationale:* {rec['financial_rationale']}")
            with col_r2:
                st.info(f"**Financial Impact:**\n\n{rec['impact']}")

# ==============================================================================
# TAB 5: WHAT-IF STRESS SIMULATOR
# ==============================================================================
with tab5:
    st.subheader("Interactive 'What-If' Scenario & Stress-Testing Simulator")
    st.caption("Simulate macroeconomic shocks, delayed collections, or operational cost cuts to observe cash runway resilience.")

    col_sim_ctrl, col_sim_view = st.columns([1, 2])

    with col_sim_ctrl:
        st.markdown("#### Scenario Controls")
        sim_sales_shock = st.slider("Revenue Shock (%)", min_value=-40, max_value=30, value=0, step=5, help="Simulate a drop or surge in customer sales.")
        sim_delay_shock = st.slider("Receivables Payment Delay (Days)", min_value=0, max_value=30, value=0, step=3, help="Simulate clients taking longer to pay.")
        sim_opex_cut = st.slider("Operational Expense Cut (%)", min_value=0, max_value=30, value=0, step=5, help="Simulate emergency reduction in overheads and supplier spending.")
        sim_discount_adoption = st.slider("Early Discount Adoption (%)", min_value=0, max_value=50, value=0, step=5, help="Simulate clients taking 2% discount to pay in the first 10 days.")

        st.markdown("---")
        st.markdown("#### Scenario Summary")
        st.write(f"- **Revenue impact:** {sim_sales_shock}%")
        st.write(f"- **Payment lag:** +{sim_delay_shock} days")
        st.write(f"- **Cost savings:** -{sim_opex_cut}%")
        st.write(f"- **Discount acceleration:** {sim_discount_adoption}% adoption")

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
            y=forecast_df["projected_cash_balance"],
            name="Baseline AI Forecast",
            mode="lines",
            line=dict(color="#94A3B8", width=2, dash="dash")
        ))
        # Simulated projection
        fig_sim.add_trace(go.Scatter(
            x=simulated_df["date"],
            y=simulated_df["projected_cash_balance"],
            name="Stress-Tested Scenario",
            mode="lines+markers",
            line=dict(color="#8B5CF6", width=3)
        ))
        # Safety line
        fig_sim.add_hline(
            y=safety_buffer,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"Safety Buffer: ₹{safety_buffer:,.0f}"
        )

        fig_sim.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Timeline",
            yaxis_title="Projected Cash Balance (INR)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Difference summary
        base_ending = forecast_df["projected_cash_balance"].iloc[-1]
        sim_ending = simulated_df["projected_cash_balance"].iloc[-1]
        diff = sim_ending - base_ending
        
        s_c1, s_c2 = st.columns(2)
        s_c1.metric("Baseline Ending Balance", f"₹{base_ending:,.0f}")
        s_c2.metric("Scenario Ending Balance", f"₹{sim_ending:,.0f}", delta=f"₹{diff:,.0f}")

st.markdown("---")
st.caption("FlowGuard • AI/ML + Finance, Law & Business (FLB) Engineering Project • Built with Streamlit, Plotly & Scikit-Learn")
