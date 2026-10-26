import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_liquidity_health(
    forecast_df: pd.DataFrame,
    safety_buffer: float = 500000.0
) -> dict:
    """
    Scans the forecasted cash flow trajectory against the company's minimum safe operating cash buffer.
    Detects potential cash shortfalls and calculates time-to-breach.
    """
    breaches = forecast_df[forecast_df["projected_cash_balance"] < safety_buffer]
    
    # Also check worst-case 95% confidence lower bound
    pessimistic_breaches = forecast_df[forecast_df["cash_balance_lower_95"] < safety_buffer]

    is_breached = len(breaches) > 0
    is_risk_detected = len(pessimistic_breaches) > 0

    min_projected = float(forecast_df["projected_cash_balance"].min())
    min_date = forecast_df.loc[forecast_df["projected_cash_balance"].idxmin(), "date"]

    if is_breached:
        first_breach_date = breaches.iloc[0]["date"]
        shortfall = safety_buffer - min_projected
        days_to_breach = (pd.to_datetime(first_breach_date) - pd.to_datetime(forecast_df["date"].iloc[0])).days + 1
        status = "CRITICAL DEFICIT ALERT"
    elif is_risk_detected:
        first_breach_date = pessimistic_breaches.iloc[0]["date"]
        shortfall = safety_buffer - float(forecast_df["cash_balance_lower_95"].min())
        days_to_breach = (pd.to_datetime(first_breach_date) - pd.to_datetime(forecast_df["date"].iloc[0])).days + 1
        status = "POTENTIAL RISK (95% CI)"
    else:
        first_breach_date = None
        shortfall = 0.0
        days_to_breach = None
        status = "HEALTHY LIQUIDITY"

    return {
        "status": status,
        "is_breached": is_breached,
        "is_risk_detected": is_risk_detected,
        "first_breach_date": first_breach_date,
        "days_to_breach": days_to_breach,
        "min_projected_balance": min_projected,
        "min_balance_date": min_date,
        "shortfall_amount": max(0.0, shortfall),
        "safety_buffer": safety_buffer
    }

def generate_recommendations(
    liquidity_status: dict,
    invoices_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> list:
    """
    Generates actionable, financially grounded prescriptive recommendations based on FLB principles:
    1. Working Capital Optimization (DSO & DPO)
    2. Dynamic AR Discounting (e.g. 2/10 Net-30)
    3. Vendor Payment Rescheduling
    4. Short-Term Line of Credit / Factoring
    5. Idle Cash Deployment (Yield Generation)
    """
    recs = []
    shortfall = liquidity_status["shortfall_amount"]

    if liquidity_status["is_breached"] or liquidity_status["is_risk_detected"]:
        # --- Lever 1: Dynamic AR Discounting ---
        # Find unpaid invoices that could be accelerated
        open_invoices = invoices_df[invoices_df["status"].isin(["Pending", "Overdue"])].copy()
        tier_a_invoices = open_invoices[open_invoices["credit_tier"] == "A"]
        
        potential_cash_unlock = tier_a_invoices["amount"].sum() * 0.98  # 2% discount
        
        if len(tier_a_invoices) > 0:
            top_clients = tier_a_invoices.groupby("client_name")["amount"].sum().head(3).to_dict()
            client_summary = ", ".join([f"{k} (INR {v:,.0f})" for k, v in top_clients.items()])
            
            recs.append({
                "type": "Accelerate Inflow (Dynamic Discounting)",
                "priority": "HIGH",
                "impact": f"Can unlock up to INR {potential_cash_unlock:,.2f} in liquid cash",
                "action": f"Offer a 2% early-payment discount for settlement within 5 days to reliable Tier-A clients: {client_summary}.",
                "financial_rationale": "Accelerating receivables reduces Days Sales Outstanding (DSO) and eliminates the need for expensive high-interest credit lines."
            })

        # --- Lever 2: Vendor Payment Re-negotiation (DPO Extension) ---
        open_expenses = expenses_df[expenses_df["status"] == "Pending"].copy()
        cogs_expenses = open_expenses[open_expenses["category"] == "COGS_Materials"]
        
        if len(cogs_expenses) > 0:
            deferrable_amount = cogs_expenses["amount"].sum()
            recs.append({
                "type": "Conserve Outflow (DPO Optimization)",
                "priority": "MEDIUM",
                "impact": f"Postpones up to INR {deferrable_amount:,.2f} in cash outflows",
                "action": "Request a 14-day payment extension on upcoming raw material shipments from key vendors, or split payments into 50% tranches.",
                "financial_rationale": "Extending Days Payable Outstanding (DPO) preserves cash during the projected shortfall window without hurting credit score."
            })

        # --- Lever 3: Working Capital Credit Line / Invoice Factoring ---
        credit_needed = round(shortfall * 1.15, -3)  # Shortfall + 15% safety margin
        recs.append({
            "type": "Emergency Liquidity Facility",
            "priority": "CRITICAL" if liquidity_status["is_breached"] else "LOW",
            "impact": f"Bridge requirement: INR {credit_needed:,.2f}",
            "action": f"Activate or apply for a pre-approved SME Working Capital Line / Invoice Factoring facility of INR {credit_needed:,.2f}.",
            "financial_rationale": "Provides a safety backstop to protect payroll and vital operating commitments while awaiting customer remittances."
        })

    else:
        # --- Surplus Cash Optimization ---
        surplus = float(forecast_df["projected_cash_balance"].min()) - liquidity_status["safety_buffer"]
        if surplus > 100000:
            deployable = surplus * 0.70  # Keep 30% buffer
            annual_yield = deployable * 0.068  # 6.8% liquid fund yield
            monthly_yield = annual_yield / 12.0
            
            recs.append({
                "type": "Treasury Optimization (Yield Generation)",
                "priority": "OPPORTUNITY",
                "impact": f"Additional annual income: ~INR {annual_yield:,.2f} (~INR {monthly_yield:,.2f}/mo)",
                "action": f"Sweep INR {deployable:,.2f} of idle cash into overnight liquid debt mutual funds or high-yield business treasury deposits.",
                "financial_rationale": "Idle cash on commercial checking accounts earns 0-2.5%. Active treasury management generates risk-free operational alpha."
            })
            
        recs.append({
            "type": "Working Capital Efficiency",
            "priority": "MAINTENANCE",
            "impact": "Maintain current DSO and healthy supplier credit terms",
            "action": "Maintain current collection cadence. Cash buffer is projected to remain healthy across the forecast horizon.",
            "financial_rationale": "Current Cash Conversion Cycle is well-balanced. Reinvest excess liquidity into inventory volume discounts."
        })

    return recs

def simulate_what_if_scenario(
    base_forecast_df: pd.DataFrame,
    sales_growth_pct: float = 0.0,
    delay_shock_days: int = 0,
    opex_reduction_pct: float = 0.0,
    discount_adoption_rate: float = 0.0
) -> pd.DataFrame:
    """
    Stress-testing engine: recalculates projected cash flows under various macroeconomic and managerial scenarios.
    - sales_growth_pct: e.g. -15.0 for a 15% revenue drop
    - delay_shock_days: customers take longer to pay
    - opex_reduction_pct: cost-cutting measures e.g. 10.0 for 10% lower overheads
    - discount_adoption_rate: % of customers taking 2% discount to pay immediately
    """
    sim_df = base_forecast_df.copy()

    # Apply sales shock
    inflow_multiplier = 1.0 + (sales_growth_pct / 100.0)
    sim_df["forecast_inflow"] = sim_df["forecast_inflow"] * inflow_multiplier

    # Apply delay shock (shifts inflow forward in time)
    if delay_shock_days > 0:
        delayed_inflow = sim_df["forecast_inflow"].shift(delay_shock_days).fillna(sim_df["forecast_inflow"].iloc[0] * 0.5)
        sim_df["forecast_inflow"] = delayed_inflow

    # Apply early payment discount acceleration
    if discount_adoption_rate > 0.0:
        # Accelerated cash brought to the first 10 days
        shift_amt = sim_df["forecast_inflow"].iloc[10:].sum() * (discount_adoption_rate / 100.0) * 0.98
        sim_df["forecast_inflow"].iloc[:10] += shift_amt / 10.0
        sim_df["forecast_inflow"].iloc[10:] *= (1.0 - (discount_adoption_rate / 100.0))

    # Apply OpEx reduction
    outflow_multiplier = 1.0 - (opex_reduction_pct / 100.0)
    sim_df["forecast_outflow"] = sim_df["forecast_outflow"] * outflow_multiplier

    # Recalculate Net Cash Flow and Ending Cash Balance
    sim_df["forecast_net_flow"] = sim_df["forecast_inflow"] - sim_df["forecast_outflow"]
    
    start_cash = float(base_forecast_df["projected_cash_balance"].iloc[0]) - float(base_forecast_df["forecast_net_flow"].iloc[0])
    sim_df["projected_cash_balance"] = start_cash + sim_df["forecast_net_flow"].cumsum()

    # Recalculate bounds with widened uncertainty for shocks
    uncertainty_scale = 1.0 + (abs(sales_growth_pct) + abs(delay_shock_days) * 2) / 100.0
    spread_80 = (base_forecast_df["cash_balance_upper_80"] - base_forecast_df["projected_cash_balance"]) * uncertainty_scale
    
    sim_df["cash_balance_lower_80"] = sim_df["projected_cash_balance"] - spread_80
    sim_df["cash_balance_upper_80"] = sim_df["projected_cash_balance"] + spread_80

    return sim_df

if __name__ == "__main__":
    from forecasting import generate_future_forecast
    ledger = pd.read_csv("data/processed/daily_cash_flow.csv")
    invoices = pd.read_csv("data/raw/invoices.csv")
    expenses = pd.read_csv("data/raw/expenses.csv")
    
    fc = generate_future_forecast(ledger, horizon_days=45, model_type="Ridge Regression")
    health = analyze_liquidity_health(fc, safety_buffer=500000.0)
    print("Liquidity Health Status:", health)
    
    recs = generate_recommendations(health, invoices, expenses, fc)
    print("\nAI Prescriptive Recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"\n{i}. [{r['priority']}] {r['type']}")
        print(f"   Action: {r['action']}")
        print(f"   Impact: {r['impact']}")
        print(f"   Financial Rationale: {r['financial_rationale']}")
