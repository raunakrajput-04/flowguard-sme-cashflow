import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

def generate_sme_dataset(
    start_date: str = "2024-01-01",
    days: int = 730,
    initial_cash: float = 450000.0,
    output_dir: str = "data"
):
    """
    Generates a realistic 2-year transactional dataset for an SME:
    1. Realistic B2B margins (COGS ~65-70% of revenue, OpEx ~20%, Net Margin ~8-12%)
    2. Working capital stress periods (Receivables delayed by 30-45 days while payables are due)
    3. Output files:
       - data/raw/invoices.csv (Accounts Receivable)
       - data/raw/expenses.csv (Accounts Payable & Operational Costs)
       - data/processed/daily_cash_flow.csv (Daily cash ledger)
    """
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start_dt + timedelta(days=i) for i in range(days)]
    end_dt = date_range[-1]

    # --- 1. Client Segments with realistic B2B payment terms ---
    clients = [
        {"name": "Apex Retailers Ltd", "credit_tier": "A", "avg_order": 45000, "delay_prob": 0.08, "terms": 30},
        {"name": "Vertex Wholesale Co", "credit_tier": "A", "avg_order": 65000, "delay_prob": 0.10, "terms": 30},
        {"name": "BlueSky Distribution", "credit_tier": "B", "avg_order": 38000, "delay_prob": 0.25, "terms": 45},
        {"name": "Horizon Tech Solutions", "credit_tier": "B", "avg_order": 42000, "delay_prob": 0.30, "terms": 45},
        {"name": "Metro Supermarkets", "credit_tier": "A", "avg_order": 80000, "delay_prob": 0.12, "terms": 60},
        {"name": "Urban Mart Network", "credit_tier": "B", "avg_order": 32000, "delay_prob": 0.28, "terms": 30},
        {"name": "Silverline Traders", "credit_tier": "C", "avg_order": 28000, "delay_prob": 0.50, "terms": 45},
        {"name": "Cascade Global Supplies", "credit_tier": "C", "avg_order": 55000, "delay_prob": 0.55, "terms": 60},
        {"name": "Zenith Commercial Corp", "credit_tier": "A", "avg_order": 58000, "delay_prob": 0.15, "terms": 30},
        {"name": "Pinnacle Hardware Hub", "credit_tier": "B", "avg_order": 35000, "delay_prob": 0.35, "terms": 45},
    ]

    # --- 2. Generate Invoices (Accounts Receivable) ---
    invoices = []
    invoice_id_counter = 1001

    for current_date in date_range:
        # Weekend volume is 40% of weekday volume
        is_weekend = current_date.weekday() >= 5
        order_prob = 0.4 if is_weekend else 0.85

        if random.random() < order_prob:
            num_orders = random.randint(1, 2) if is_weekend else random.randint(1, 3)
            
            # Seasonal factor: high in Q4 (festive/year-end), dips in summer
            month = current_date.month
            seasonality = 1.25 if month in [10, 11, 12] else (0.85 if month in [4, 5] else 1.0)
            
            # Special liquidity crunch scenario: Around day 500 to 560, clients pay significantly later!
            days_since_start = (current_date - start_dt).days
            stress_period = 500 <= days_since_start <= 560

            for _ in range(num_orders):
                client = random.choice(clients)
                order_amount = round(np.random.normal(client["avg_order"], client["avg_order"] * 0.12) * seasonality, 2)
                order_amount = max(order_amount, 8000.0)

                terms = client["terms"]
                due_date = current_date + timedelta(days=terms)

                # Delay logic
                delay_chance = min(0.9, client["delay_prob"] * (2.2 if stress_period else 1.0))
                is_delayed = random.random() < delay_chance

                if is_delayed:
                    delay_days = random.randint(10, 50) if stress_period else random.randint(5, 25)
                else:
                    # Early settlement incentive (1-5 days early)
                    delay_days = random.randint(-4, 0)

                settlement_dt = due_date + timedelta(days=delay_days)

                if settlement_dt <= end_dt:
                    status = "Paid"
                    # 2% cash discount if settled >= 3 days early
                    collected = round(order_amount * 0.98 if delay_days <= -3 else order_amount, 2)
                    settlement_str = settlement_dt.strftime("%Y-%m-%d")
                else:
                    collected = 0.0
                    settlement_str = None
                    status = "Overdue" if due_date < end_dt else "Pending"

                invoices.append({
                    "invoice_id": f"INV-{invoice_id_counter}",
                    "client_name": client["name"],
                    "credit_tier": client["credit_tier"],
                    "issue_date": current_date.strftime("%Y-%m-%d"),
                    "due_date": due_date.strftime("%Y-%m-%d"),
                    "payment_terms": f"Net-{terms}",
                    "amount": order_amount,
                    "status": status,
                    "settlement_date": settlement_str,
                    "collected_amount": collected,
                    "days_delayed": delay_days if status == "Paid" else max(0, (end_dt - due_date).days if status == "Overdue" else 0)
                })
                invoice_id_counter += 1

    invoices_df = pd.DataFrame(invoices)

    # --- 3. Generate Expenses (Accounts Payable & Operations) ---
    expenses = []
    expense_id_counter = 5001

    for current_date in date_range:
        day_of_month = current_date.day
        month = current_date.month

        # Fixed Rent on the 1st of every month: INR 65,000
        if day_of_month == 1:
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "Metro Industrial Properties",
                "category": "Rent",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": current_date.strftime("%Y-%m-%d"),
                "amount": 65000.0,
                "payment_date": current_date.strftime("%Y-%m-%d"),
                "status": "Paid"
            })
            expense_id_counter += 1

        # Fixed Payroll on the 28th of every month: INR 180,000
        if day_of_month == 28:
            payroll = round(np.random.normal(180000, 4000), 2)
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "Staff Payroll Reserve",
                "category": "Payroll",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": current_date.strftime("%Y-%m-%d"),
                "amount": payroll,
                "payment_date": current_date.strftime("%Y-%m-%d"),
                "status": "Paid"
            })
            expense_id_counter += 1

        # Quarterly Advance Corporate Tax on the 15th (March, June, Sept, Dec): INR 50,000
        if month in [3, 6, 9, 12] and day_of_month == 15:
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "Treasury & Tax Authority",
                "category": "Taxation",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": current_date.strftime("%Y-%m-%d"),
                "amount": round(np.random.normal(52000, 3000), 2),
                "payment_date": current_date.strftime("%Y-%m-%d"),
                "status": "Paid"
            })
            expense_id_counter += 1

        # Regular COGS / Raw Materials Procurement (2 to 3 times per week, ~65% of sales value)
        if current_date.weekday() in [0, 2, 4]:
            cogs_amount = round(np.random.normal(32000, 5000), 2)
            # Net-30 vendor payment terms
            due = current_date + timedelta(days=30)
            pay_date = due if due <= end_dt else None
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "National Raw Materials Corp",
                "category": "COGS_Materials",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": due.strftime("%Y-%m-%d"),
                "amount": cogs_amount,
                "payment_date": pay_date.strftime("%Y-%m-%d") if pay_date else None,
                "status": "Paid" if pay_date else "Pending"
            })
            expense_id_counter += 1

        # Logistics & Distribution (every Tuesday & Friday)
        if current_date.weekday() in [1, 4]:
            log_amount = round(np.random.normal(7500, 1000), 2)
            due = current_date + timedelta(days=15)
            pay_date = due if due <= end_dt else None
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "FastTrack Freight & Logistics",
                "category": "Logistics",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": due.strftime("%Y-%m-%d"),
                "amount": log_amount,
                "payment_date": pay_date.strftime("%Y-%m-%d") if pay_date else None,
                "status": "Paid" if pay_date else "Pending"
            })
            expense_id_counter += 1

        # Utilities & Operations (Mid-month, 10th)
        if day_of_month == 10:
            expenses.append({
                "expense_id": f"EXP-{expense_id_counter}",
                "vendor_name": "State Power & Municipal Corp",
                "category": "Utilities",
                "incurred_date": current_date.strftime("%Y-%m-%d"),
                "due_date": current_date.strftime("%Y-%m-%d"),
                "amount": round(np.random.normal(12500, 1200), 2),
                "payment_date": current_date.strftime("%Y-%m-%d"),
                "status": "Paid"
            })
            expense_id_counter += 1

    expenses_df = pd.DataFrame(expenses)

    # --- 4. Build Daily Cash Flow Ledger ---
    paid_invoices = invoices_df[invoices_df["status"] == "Paid"].copy()
    daily_inflow = paid_invoices.groupby("settlement_date")["collected_amount"].sum().reset_index()
    daily_inflow.rename(columns={"settlement_date": "date", "collected_amount": "inflow"}, inplace=True)

    paid_expenses = expenses_df[expenses_df["status"] == "Paid"].copy()
    daily_outflow = paid_expenses.groupby("payment_date")["amount"].sum().reset_index()
    daily_outflow.rename(columns={"payment_date": "date", "amount": "outflow"}, inplace=True)

    calendar_df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in date_range]})
    ledger = calendar_df.merge(daily_inflow, on="date", how="left").fillna(0.0)
    ledger = ledger.merge(daily_outflow, on="date", how="left").fillna(0.0)

    ledger["net_cash_flow"] = ledger["inflow"] - ledger["outflow"]
    ledger["cash_balance"] = initial_cash + ledger["net_cash_flow"].cumsum()

    # Save to CSV
    invoices_path = os.path.join(output_dir, "raw", "invoices.csv")
    expenses_path = os.path.join(output_dir, "raw", "expenses.csv")
    ledger_path = os.path.join(output_dir, "processed", "daily_cash_flow.csv")

    invoices_df.to_csv(invoices_path, index=False)
    expenses_df.to_csv(expenses_path, index=False)
    ledger.to_csv(ledger_path, index=False)

    print(f"Generated {len(invoices_df)} invoices -> {invoices_path}")
    print(f"Generated {len(expenses_df)} expenses -> {expenses_path}")
    print(f"Generated {len(ledger)} daily cash records -> {ledger_path}")
    print(f"Initial Cash: INR {initial_cash:,.2f}")
    print(f"Final Cash Balance: INR {ledger['cash_balance'].iloc[-1]:,.2f}")
    print(f"Min Cash Balance during 2 years: INR {ledger['cash_balance'].min():,.2f}")
    print(f"Max Cash Balance: INR {ledger['cash_balance'].max():,.2f}")

    return invoices_df, expenses_df, ledger

if __name__ == "__main__":
    generate_sme_dataset()
