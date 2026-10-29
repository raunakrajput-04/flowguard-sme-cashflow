import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_flowguard_deck(output_path="FlowGuard_Presentation.pptx"):
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette: Clean Executive Navy / Slate Theme
    NAVY = RGBColor(15, 23, 42)         # #0F172A
    CARD_BG = RGBColor(248, 250, 252)   # #F8FAFC
    CARD_BORDER = RGBColor(226, 232, 240)
    DARK_BLUE = RGBColor(30, 58, 138)   # #1E3A8A
    ACCENT_CYAN = RGBColor(2, 132, 199) # #0284C7
    TEXT_MAIN = RGBColor(15, 23, 42)
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748B
    GREEN = RGBColor(16, 185, 129)      # #10B981
    RED = RGBColor(239, 68, 68)         # #EF4444
    WHITE = RGBColor(255, 255, 255)

    def add_header(slide, title_text, category_text="FLOWGUARD • FLB & AI/ML ENGINEERING PROJECT"):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, title, body_bullets, accent_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = accent_color if accent_color else CARD_BORDER
        shape.line.width = Pt(1.5 if accent_color else 1)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.25)

        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(16)
        p_title.font.bold = True
        p_title.font.color.rgb = DARK_BLUE
        p_title.space_after = Pt(12)

        for bullet in body_bullets:
            p = tf.add_paragraph()
            p.text = "• " + bullet
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(8)

    # ==========================================
    # SLIDE 1: TITLE SLIDE (Cover)
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Title Box
    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.5))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "FINANCE, LAW & BUSINESS (FLB) + AI/ML PROJECT"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = RGBColor(56, 189, 248)
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    p_main.text = "FlowGuard: SME Cash Flow Forecasting & Working Capital Decision Support System"
    p_main.font.size = Pt(32)
    p_main.font.bold = True
    p_main.font.color.rgb = WHITE
    p_main.space_after = Pt(16)

    p_sub = tf1.add_paragraph()
    p_sub.text = "An intelligent decision-support system that predicts daily cash balances, detects liquidity crunches early, and provides actionable working capital solutions."
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Presented by: Raunak Rajput | AIML Department"
    p_meta.font.size = Pt(14)
    p_meta.font.bold = True
    p_meta.font.color.rgb = WHITE

    # ==========================================
    # SLIDE 2: THE BUSINESS PROBLEM
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Real-World Problem: Why Profitable SMEs Go Bankrupt")
    
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             "1. The Fatal Cash Paradox", [
                 "82% of small and medium businesses (SMEs) fail due to cash flow mismanagement, NOT lack of sales.",
                 "Profit is on paper (accrual accounting), but salaries and rent must be paid with real bank cash.",
                 "A company can show ₹50 Lakhs in sales, but if clients haven't paid, the business cannot survive next month.",
                 "Key insight: 'Revenue is vanity, profit is sanity, but cash is king.'"
             ], ACCENT_CYAN)

    add_card(slide2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), 
             "2. The Timing Mismatch Nightmare", [
                 "Accounts Receivable (Inflows): Clients take 30 to 60 days to pay their invoices, often with unexpected delays.",
                 "Accounts Payable (Outflows): Suppliers, payroll (28th of every month), rent, and taxes demand prompt settlement.",
                 "Result: A dangerous gap where bank balances drop close to zero without warning.",
                 "Goal of FlowGuard: Give managers a 30-to-90 day advance radar before cash dries up."
             ], RED)

    # ==========================================
    # SLIDE 3: FINANCIAL CONCEPTS MADE SIMPLE
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Core Financial Terms Made Simple (FLB Concepts)")

    add_card(slide3, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "Working Capital & Runway", [
                 "Working Capital = Current Assets minus Current Liabilities.",
                 "It is the daily oxygen buffer of a company.",
                 "Current Ratio: Benchmark is > 1.5x (measures short-term safety).",
                 "Cash Runway: Number of months the company can survive at its current monthly burn rate."
             ])

    add_card(slide3, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "DSO & DPO Cycles", [
                 "DSO (Days Sales Outstanding): Average number of days customers take to pay you. Lower is better!",
                 "DPO (Days Payable Outstanding): Average number of days you take to pay suppliers.",
                 "DIO (Days Inventory Outstanding): Days items sit in warehouse before selling."
             ])

    add_card(slide3, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8), 
             "Cash Conversion Cycle (CCC)", [
                 "Formula: CCC = DIO + DSO - DPO",
                 "In plain English: The number of days it takes for ₹1 spent on raw materials to come back into your bank account as cash.",
                 "Shorter CCC = Faster cash velocity and less need for bank loans!"
             ], GREEN)

    # ==========================================
    # SLIDE 4: WHY TRADITIONAL EXCEL FAILS
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Why Traditional Excel & Spreadsheets Fail for SMEs")

    add_card(slide4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             "Limitations of Excel Forecasting", [
                 "Static Assumptions: Spreadsheets assume all clients pay exactly on the due date. In reality, they rarely do.",
                 "No Pattern Recognition: Excel cannot automatically detect that 'Client X always delays payments by 18 days in winter'.",
                 "Reactive, not Proactive: Business owners only see problems when a cheque bounces or payroll is tomorrow.",
                 "High manual effort: Easy to make formula mistakes in large multi-sheet workbooks."
             ], RED)

    add_card(slide4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), 
             "How FlowGuard Solves This", [
                 "Automated Daily Ledger: Integrates past invoices, payments, and expenses continuously.",
                 "Probabilistic Forecasting: Learns client behavior patterns and payment delays from historical data.",
                 "Uncertainty Bands: Displays 80% and 95% confidence intervals (Best case vs Worst case scenario).",
                 "Interactive What-If Testing: Recalculates cash runway in 0.2 seconds when sliders are moved."
             ], GREEN)

    # ==========================================
    # SLIDE 5: HOW THE AI WORKS (Plain English)
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "How FlowGuard's AI Works (Demystified in Plain English)")

    add_card(slide5, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "1. Learns Habits", [
                 "Tracks historical customer payments over 2 years (730 days).",
                 "Identifies client reliability tiers (Tier A pays on time, Tier C delays 20+ days).",
                 "Considers day-of-week effects (B2B payments drop on weekends)."
             ])

    add_card(slide5, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "2. Tracks Calendar Clocks", [
                 "Captures payroll cycles (fixed salaries disbursed every 28th).",
                 "Tracks rent on the 1st of every month.",
                 "Models quarterly corporate advance taxes (March, June, Sept, Dec 15th).",
                 "Accounts for festive quarter surges."
             ])

    add_card(slide5, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8), 
             "3. Multi-Step Forecast", [
                 "Simultaneously forecasts daily Inflows (collections) and Outflows (expenses).",
                 "Computes Net Cash Flow = Inflow - Outflow.",
                 "Projects daily bank ending balance up to 90 days ahead with risk bands."
             ], ACCENT_CYAN)

    # ==========================================
    # SLIDE 6: AI BENCHMARKING (Engineering Rigor)
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Model Benchmarking: ML Regressors vs Statistical Baseline")

    add_card(slide6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             "Algorithms Evaluated", [
                 "1. Baseline: Holt-Winters Exponential Smoothing (classical statistical benchmark).",
                 "2. Ridge Regression: Regularized linear model preventing overfitting on lag features.",
                 "3. Random Forest: Ensemble decision trees capturing non-linear interactions.",
                 "4. Gradient Boosting (GBM): Sequentially trained decision trees optimizing prediction residuals."
             ])

    add_card(slide6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), 
             "Validation Results & Metrics", [
                 "Evaluated on a 60-day holdout validation dataset.",
                 "Metrics Used: MAE (Mean Absolute Error), RMSE, and MAPE.",
                 "Key finding: Machine learning models with lag features (t-1, t-7, t-30) and payroll indicators outperformed classical baseline smoothing.",
                 "Why it matters: Eliminates unexpected cash surprises for CFOs and owners."
             ], GREEN)

    # ==========================================
    # SLIDE 7: PRESCRIPTIVE ADVISOR (Not Just Predictions, Solutions!)
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Prescriptive Decision Support: 4 Actionable Financial Levers")

    add_card(slide7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.3), 
             "Lever 1: Dynamic AR Discounting", [
                 "Action: Offer 2% discount (e.g. 2/10 Net 30) to Tier-A clients.",
                 "Impact: Unlocks immediate cash collections within 5 days to avert deficit."
             ], GREEN)

    add_card(slide7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.3), 
             "Lever 2: Vendor DPO Negotiation", [
                 "Action: Request 14-day payment extension on raw materials.",
                 "Impact: Conserves outflow without hurting commercial supplier trust."
             ], ACCENT_CYAN)

    add_card(slide7, Inches(0.8), Inches(4.4), Inches(5.6), Inches(2.3), 
             "Lever 3: Working Capital Credit Line", [
                 "Action: Sizes the exact emergency credit line (Shortfall + 15% cushion).",
                 "Impact: Prevents taking expensive, unnecessary bank debt."
             ], RED)

    add_card(slide7, Inches(6.8), Inches(4.4), Inches(5.7), Inches(2.3), 
             "Lever 4: Idle Cash Yield Sweep", [
                 "Action: If cash is surplus, sweep excess funds into overnight mutual funds.",
                 "Impact: Generates 6.8% risk-free annualized treasury returns."
             ])

    # ==========================================
    # SLIDE 8: THE WHAT-IF STRESS SIMULATOR
    # ==========================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Interactive 'What-If' Stress-Testing Simulator")

    add_card(slide8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             "Real-Time Managerial Testing", [
                 "Business managers can test extreme scenarios with 1-click presets:",
                 "🚀 Aggressive Expansion: +25% Sales, 30% discount adoption.",
                 "⚠️ Client Payment Freeze: -15% Sales, +15 days payment delay.",
                 "🚨 Macro Recession Shock: -35% Sales, +25 days delay, -15% emergency OpEx cuts.",
                 "Instant Feedback: Recalculates projected cash balance curve and runway live."
             ], ACCENT_CYAN)

    add_card(slide8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), 
             "Business Value of Stress Testing", [
                 "Pre-Mortem Analysis: Test your business's survival BEFORE entering a crisis.",
                 "Bank Negotiation: Show lenders exact stress-tested cash flow forecasts to secure credit lines faster.",
                 "Confidence: Management knows exactly which cost-cutting levers save the company during a downturn."
             ], GREEN)

    # ==========================================
    # SLIDE 9: SYSTEM ARCHITECTURE & TECH STACK
    # ==========================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "System Architecture & Engineering Implementation")

    add_card(slide9, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "Frontend & UI", [
                 "Streamlit Framework",
                 "Bloomberg / TradingView Executive Dark Theme",
                 "Plotly Interactive Fan Charts & Gauges",
                 "Multi-Currency Toggle (INR, USD, EUR)"
             ])

    add_card(slide9, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8), 
             "Intelligence Engine", [
                 "Scikit-Learn (Gradient Boosting, Random Forest, Ridge)",
                 "Statsmodels (Holt-Winters Baseline)",
                 "Feature Engineering (Lags, rolling stats, Fourier seasonality)",
                 "Heuristic Prescriptive Rule Engine"
             ])

    add_card(slide9, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8), 
             "Data Pipeline & Code", [
                 "Realistic 730-day SME Transactional Ledger",
                 "Accounts Receivable & Accounts Payable generator",
                 "Version Controlled on GitHub",
                 "1-Click Executive Report Export (.TXT)"
             ], ACCENT_CYAN)

    # ==========================================
    # SLIDE 10: CONCLUSION & VIVA READINESS
    # ==========================================
    slide10 = prs.slides.add_slide(blank_layout)
    bg10 = slide10.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg10.fill.solid()
    bg10.fill.fore_color.rgb = NAVY
    bg10.line.fill.background()

    c_box = slide10.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(10.9), Inches(5.2))
    tfc = c_box.text_frame
    tfc.word_wrap = True

    p_c_title = tfc.paragraphs[0]
    p_c_title.text = "Key Takeaways & Conclusion"
    p_c_title.font.size = Pt(28)
    p_c_title.font.bold = True
    p_c_title.font.color.rgb = WHITE
    p_c_title.space_after = Pt(20)

    bullets = [
        "1. Bridges the Gap: Combines rigorous corporate finance (FLB) with practical machine learning (AIML).",
        "2. Solves Real Problem: Protects SMEs from sudden liquidity death through 30-90 day advance radar.",
        "3. Understandable AI: Avoids overly complex 'black-box' models; uses explainable time-series patterns and clear financial ratios.",
        "4. Actionable: Doesn't just predict problems—provides 4 prescriptive financial solutions and stress-testing tools.",
        "5. Fully Functional: Live interactive web app + open-source repository on GitHub."
    ]

    for b in bullets:
        p = tfc.add_paragraph()
        p.text = b
        p.font.size = Pt(16)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_after = Pt(12)

    p_qa = tfc.add_paragraph()
    p_qa.text = "\nThank You! Open for Questions & Demonstration."
    p_qa.font.size = Pt(20)
    p_qa.font.bold = True
    p_qa.font.color.rgb = RGBColor(56, 189, 248)

    prs.save(output_path)
    print(f"Presentation generated successfully: {output_path}")

if __name__ == "__main__":
    create_flowguard_deck()
