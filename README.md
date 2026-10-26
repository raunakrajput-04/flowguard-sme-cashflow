# FlowGuard: AI-Driven SME Cash Flow Forecasting & Working Capital Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Stats-Statsmodels-green.svg)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interdisciplinary **Artificial Intelligence & Machine Learning (AIML)** and **Finance, Law & Business (FLB)** engineering project designed to solve liquidity volatility and working capital constraints for Small and Medium Enterprises (SMEs).

---

## 📌 Executive Summary

> *"Revenue is vanity, profit is sanity, but cash is king."*

According to global small business studies, **over 82% of SMEs fail due to cash flow mismanagement**, even while remaining profitable on an accrual accounting basis. The primary cause is the mismatch between **Accounts Receivable (AR)** collections and **Accounts Payable (AP)** obligations:
- Cash is trapped in unpaid client invoices (30–60 day payment terms).
- Fixed operational expenditures (payroll, rent, debt servicing, utility bills) must be settled on strict deadlines.

**FlowGuard** bridges financial economics with machine learning by providing:
1. **Multi-Step Recursive Cash Forecasting** (30, 45, 60, and 90-day daily horizons) with **80% and 95% Confidence Intervals**.
2. **Automated Working Capital & Liquidity Diagnostics** (DSO, DPO, DIO, Cash Conversion Cycle, Current & Quick Ratios).
3. **AI Prescriptive Treasury Advisory**: Translates quantitative predictions into actionable business interventions (dynamic discounting, vendor payment rescheduling, credit line sizing, and idle cash yield sweeps).
4. **Interactive Stress-Testing Simulator**: Dynamic "What-If" scenario modeling for revenue shocks, payment delays, and OpEx adjustments.

---

## 🏛️ Business & Financial Foundations (FLB Angle)

The system computes and evaluates standard corporate finance and managerial accounting metrics:

### 1. Working Capital & Liquidity Health
$$\text{Net Working Capital} = \text{Current Assets} - \text{Current Liabilities}$$
$$\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}} \quad (\text{Target: } 1.5 - 2.0)$$
$$\text{Quick Ratio (Acid-Test)} = \frac{\text{Cash} + \text{Marketable Securities} + \text{Accounts Receivable}}{\text{Current Liabilities}} \quad (\text{Target: } > 1.0)$$

### 2. Operating Cycle & Cash Conversion Cycle (CCC)
$$\text{DSO (Days Sales Outstanding)} = \left(\frac{\text{Accounts Receivable}}{\text{Total Credit Sales}}\right) \times \text{Days in Period}$$
$$\text{DPO (Days Payable Outstanding)} = \left(\frac{\text{Accounts Payable}}{\text{Cost of Goods Sold}}\right) \times \text{Days in Period}$$
$$\text{DIO (Days Inventory Outstanding)} = \left(\frac{\text{Average Inventory}}{\text{Cost of Goods Sold}}\right) \times \text{Days in Period}$$
$$\mathbf{CCC} = \mathbf{DIO} + \mathbf{DSO} - \mathbf{DPO}$$

* **Economic Interpretation:** A shorter or negative CCC indicates higher capital efficiency, reducing the SME's dependency on expensive external working capital debt.

### 3. Net Burn Rate & Cash Runway
$$\text{Net Cash Flow}_t = \text{Inflow}_t - \text{Outflow}_t$$
$$\text{Cash Runway (Months)} = \frac{\text{Current Liquid Cash Balance}}{\text{Average Monthly Outflows}}$$

---

## 🧠 AI & Machine Learning Architecture

```
Raw B2B Invoices & Expenses Ledger (730 Days)
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│               Feature Engineering Pipeline             │
│  - Autoregressive Lags: t-1, t-2, t-7, t-14, t-28, t-30│
│  - Rolling Windows: 7d/14d/30d Mean & 7d Volatility    │
│  - Calendar Cycles: Day-of-Week, Month-End, Payroll    │
│  - Smooth Seasonality: Annual & Weekly Fourier Terms   │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│             Multi-Model Benchmarking Engine            │
│  1. Baseline: Holt-Winters Exponential Smoothing       │
│  2. Regularized Linear: Ridge Regression (L2)          │
│  3. Ensemble Trees: Random Forest Regressor            │
│  4. Gradient Boosting Regressor (GBM)                  │
│     * Evaluation Metrics: MAE, RMSE, MAPE              │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│         Multi-Step Recursive Cash Flow Forecaster      │
│  - Simultaneous Inflow & Outflow Prediction            │
│  - Dynamic Cumulative Ending Balance Trajectory        │
│  - 80% & 95% Parametric Confidence Bands               │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│        Prescriptive AI Liquidity & Treasury Advisor    │
│  - Liquidity Shortfall & Grace Period Detector         │
│  - Dynamic Early-Payment Discounting (2/10 Net-30)     │
│  - Vendor Payment Term Renegotiation (DPO Extension)   │
│  - Working Capital Credit Line Sizing                  │
│  - Idle Cash Treasury Yield Sweep (Overnight Mutuals)  │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│         Interactive Streamlit & Plotly Dashboard       │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```
flowguard/
│
├── data/
│   ├── raw/
│   │   ├── invoices.csv           # B2B customer invoices with credit tiers & payment terms
│   │   └── expenses.csv           # Vendor bills, rent, payroll, tax & operating disbursements
│   └── processed/
│       └── daily_cash_flow.csv    # 2-year daily aggregated cash ledger
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Realistic 730-day SME financial transaction generator
│   ├── financial_metrics.py       # Working capital, DSO, DPO, CCC & AR aging analytics
│   ├── forecasting.py             # Feature engineering, ML models, evaluation & recursive forecast
│   └── advisor_engine.py          # Liquidity risk scanner, prescriptive rules & what-if simulator
│
├── app.py                         # Production-ready Streamlit interactive application
├── requirements.txt               # Project dependencies
└── README.md                      # Academic documentation & viva presentation guide
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/raunakrajput-04/flowguard-sme-cashflow.git
cd flowguard-sme-cashflow

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (if needed)
```bash
python src/data_generator.py
```

### 3. Launch the FlowGuard Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application.

---

## 🖥️ Dashboard Walkthrough

The web application is partitioned into 5 functional modules:

1. **Executive KPI Dashboard:**
   - Displays real-time Current Cash Balance, Working Capital, Cash Conversion Cycle, and Runway.
   - Monthly historical Inflow vs Outflow cash flow waterfall.
2. **AI Cash Flow Forecasting:**
   - Multi-step daily cash balance trajectory with 80% and 95% uncertainty bands.
   - Red threshold indicator for minimum safe operating balance.
   - Model benchmarking table comparing Ridge, Random Forest, Gradient Boosting, and Holt-Winters.
3. **Working Capital & AR/AP Aging:**
   - Donut chart of Accounts Receivable Aging buckets (Current, 1–30d, 31–60d, 61–90d, 90+d overdue).
   - Client credit tier risk segmentation (Tier A vs B vs C average delay days).
   - High-priority open invoices table.
4. **AI Liquidity Advisor:**
   - Automated liquidity status alert (Healthy / Volatility Warning / Critical Deficit).
   - Prescriptive action cards with quantitative financial impact analysis (Dynamic discounting, DPO optimization, credit facility requirements, idle cash yield sweep).
5. **Interactive What-If Stress Simulator:**
   - Sliders for Revenue Shock ($\pm 30\%$), Collection Delay ($+0 \text{ to } 30\text{ days}$), OpEx Cost Cutting, and Early Payment Discount Adoption.
   - Real-time comparison between Baseline AI Forecast vs Stress-Tested Scenario.

---

## 🎓 Viva / Evaluation Presentation Guide (FLB + AIML)

### Q1: Why is Cash Flow Forecasting more critical than Net Profit for an SME?
**Answer:**
Accrual accounting records revenues when invoices are issued and expenses when incurred, regardless of actual cash receipt. An SME can report high paper profits while facing bankruptcy if customers delay payment by 60 days and the company lacks liquid cash to pay payroll or suppliers. Cash flow forecasting monitors actual liquidity velocity.

### Q2: How does your ML pipeline handle the temporal nature of financial transactions?
**Answer:**
We construct autoregressive lag features ($t-1, t-2, t-7, t-14, t-28, t-30$), rolling statistical windows (7-day mean and volatility), calendar indicators (business days, month-end, payroll cycles), and sinusoidal Fourier terms for annual seasonality. Multi-step forecasts are generated recursively so predicted values feed back into subsequent lag calculations.

### Q3: What is the Cash Conversion Cycle (CCC) and why does FlowGuard track it?
**Answer:**
$\text{CCC} = \text{DIO} + \text{DSO} - \text{DPO}$. It tracks the net days required to convert operational investments (raw materials and inventory) back into realized cash collections. A high CCC means working capital is locked up; shortening DSO via dynamic discounting or extending DPO lowers capital requirements.

### Q4: How are confidence intervals generated for the cash balance forecast?
**Answer:**
We compute the standard deviation of residuals ($\sigma_{\text{net}}$) over historical out-of-sample periods. Uncertainty scales with the square root of forecast horizon ($\sqrt{h}$):
$$\text{Bound}_{95\%} = \hat{y}_t \pm 1.96 \cdot \sigma_{\text{net}} \sqrt{\frac{t}{7}}$$

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
