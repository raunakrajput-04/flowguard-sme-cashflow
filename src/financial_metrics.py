import pandas as pd
import numpy as np
from datetime import datetime

def calculate_kpis(
    invoices_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    as_of_date: str = None
) -> dict:
    """
    Computes key financial ratios and working capital metrics for the SME:
    - Days Sales Outstanding (DSO)
    - Days Payable Outstanding (DPO)
    - Cash Conversion Cycle (CCC)
    - Working Capital & Liquidity Ratios (Current & Quick Ratio)
    - Cash Runway & Monthly Burn Rate
    - Accounts Receivable Aging Buckets
    """
    if as_of_date is None:
        as_of_date = ledger_df["date"].max()

    as_of_dt = pd.to_datetime(as_of_date)
    
    # Filter historical data up to as_of_date
    hist_ledger = ledger_df[pd.to_datetime(ledger_df["date"]) <= as_of_dt].copy()
    current_cash = float(hist_ledger["cash_balance"].iloc[-1]) if not hist_ledger.empty else 0.0

    # 1. Accounts Receivable (AR) & Aging
    invoices_sub = invoices_df[pd.to_datetime(invoices_df["issue_date"]) <= as_of_dt].copy()
    
    # An invoice is unsettled as of this date if settlement_date is null or > as_of_date
    unsettled_mask = (
        (invoices_sub["settlement_date"].isna()) | 
        (pd.to_datetime(invoices_sub["settlement_date"]) > as_of_dt)
    )
    ar_df = invoices_sub[unsettled_mask].copy()
    total_ar = float(ar_df["amount"].sum())

    # AR Aging Buckets based on due_date
    ar_df["due_dt"] = pd.to_datetime(ar_df["due_date"])
    ar_df["days_overdue"] = (as_of_dt - ar_df["due_dt"]).dt.days

    aging_buckets = {
        "Current (0-30 days)": float(ar_df[ar_df["days_overdue"] <= 0]["amount"].sum()),
        "1-30 Days Past Due": float(ar_df[(ar_df["days_overdue"] > 0) & (ar_df["days_overdue"] <= 30)]["amount"].sum()),
        "31-60 Days Past Due": float(ar_df[(ar_df["days_overdue"] > 30) & (ar_df["days_overdue"] <= 60)]["amount"].sum()),
        "61-90 Days Past Due": float(ar_df[(ar_df["days_overdue"] > 60) & (ar_df["days_overdue"] <= 90)]["amount"].sum()),
        "90+ Days Past Due": float(ar_df[ar_df["days_overdue"] > 90]["amount"].sum()),
    }

    # 2. Accounts Payable (AP)
    expenses_sub = expenses_df[pd.to_datetime(expenses_df["incurred_date"]) <= as_of_dt].copy()
    unpaid_exp_mask = (
        (expenses_sub["payment_date"].isna()) | 
        (pd.to_datetime(expenses_sub["payment_date"]) > as_of_dt)
    )
    ap_df = expenses_sub[unpaid_exp_mask].copy()
    total_ap = float(ap_df["amount"].sum())

    # 3. Trailing 90-Day Performance for Ratios
    t90_start = as_of_dt - pd.Timedelta(days=90)
    t90_ledger = hist_ledger[pd.to_datetime(hist_ledger["date"]) >= t90_start]
    
    total_sales_t90 = float(invoices_sub[pd.to_datetime(invoices_sub["issue_date"]) >= t90_start]["amount"].sum())
    total_sales_t90 = max(total_sales_t90, 1.0)
    
    cogs_t90 = float(expenses_sub[
        (pd.to_datetime(expenses_sub["incurred_date"]) >= t90_start) & 
        (expenses_sub["category"] == "COGS_Materials")
    ]["amount"].sum())
    cogs_t90 = max(cogs_t90, 1.0)

    # DSO = (Accounts Receivable / Total Credit Sales) * 90
    dso = round((total_ar / total_sales_t90) * 90, 1)

    # DPO = (Accounts Payable / COGS) * 90
    dpo = round((total_ap / cogs_t90) * 90, 1)

    # DIO benchmark for industry
    dio = 38.0

    # Cash Conversion Cycle (CCC) = DIO + DSO - DPO
    ccc = round(dio + dso - dpo, 1)

    # 4. Working Capital & Liquidity Ratios
    estimated_inventory = cogs_t90 * (dio / 90.0)
    current_assets = current_cash + total_ar + estimated_inventory
    current_liabilities = total_ap + 180000.0  # AP + upcoming accrued monthly payroll/taxes
    
    working_capital = current_assets - current_liabilities
    current_ratio = round(current_assets / max(current_liabilities, 1.0), 2)
    quick_ratio = round((current_cash + total_ar) / max(current_liabilities, 1.0), 2)

    # 5. Burn Rate & Runway
    avg_monthly_outflow = float(t90_ledger["outflow"].sum()) / 3.0 if len(t90_ledger) > 0 else 250000.0
    avg_monthly_inflow = float(t90_ledger["inflow"].sum()) / 3.0 if len(t90_ledger) > 0 else 300000.0
    net_monthly_burn = avg_monthly_outflow - avg_monthly_inflow

    runway_months = round(current_cash / max(avg_monthly_outflow, 1.0), 1)

    return {
        "as_of_date": as_of_date,
        "current_cash": current_cash,
        "total_ar": total_ar,
        "total_ap": total_ap,
        "working_capital": working_capital,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "dso_days": dso,
        "dpo_days": dpo,
        "dio_days": dio,
        "ccc_days": ccc,
        "avg_monthly_outflow": avg_monthly_outflow,
        "avg_monthly_inflow": avg_monthly_inflow,
        "net_monthly_burn": net_monthly_burn,
        "runway_months": runway_months,
        "aging_buckets": aging_buckets,
        "num_unpaid_invoices": len(ar_df),
        "num_unpaid_bills": len(ap_df)
    }

if __name__ == "__main__":
    invoices = pd.read_csv("data/raw/invoices.csv")
    expenses = pd.read_csv("data/raw/expenses.csv")
    ledger = pd.read_csv("data/processed/daily_cash_flow.csv")
    metrics = calculate_kpis(invoices, expenses, ledger)
    print("--- SME FINANCIAL KPIs ---")
    for k, v in metrics.items():
        if k != "aging_buckets":
            print(f"{k}: {v}")
    print("\nAR Aging:")
    for b, val in metrics["aging_buckets"].items():
        print(f"  {b}: INR {val:,.2f}")
