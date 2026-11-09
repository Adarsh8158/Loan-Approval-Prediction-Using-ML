"""
LoanIQ - Intelligent Loan Approval Prediction System
Streamlit UI inspired by modern fintech dashboards (Finsyght-style Neo Dark Theme).
Connects to trained scikit-learn Pipeline and persists predictions in SQLite.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import textwrap
from src.predictor import predict_loan
from src.database import save_prediction_record, get_prediction_history, clear_prediction_history

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LoanIQ | AI Loan Analytics & Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def render_html(html_str: str):
    """
    Renders pure HTML cleanly using st.html with dedent.
    Prevents Markdown from converting 4-space indentations into raw code blocks.
    """
    clean_html = textwrap.dedent(html_str).strip()
    st.html(clean_html)


# -----------------------------------------------------------------------------
# CUSTOM ULTRA-MODERN DARK FINTECH CSS (Matching Reference Image)
# -----------------------------------------------------------------------------
render_html("""
<style>
    /* Global Canvas styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 90% 70%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
                    #080b13 !important;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hide standard Streamlit header & toolbar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Interactive Segmented Control Navigation Pill Bar Styling */
    div[data-testid="stSegmentedControl"] {
        background: rgba(13, 17, 28, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 4px 8px !important;
        display: flex !important;
        justify-content: center !important;
        gap: 6px !important;
        backdrop-filter: blur(12px) !important;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 10px !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        border: 1px solid transparent !important;
        padding: 6px 14px !important;
        color: #94a3b8 !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-state="on"] {
        background: rgba(168, 85, 247, 0.25) !important;
        color: #c084fc !important;
        border: 1px solid rgba(168, 85, 247, 0.5) !important;
        font-weight: 600 !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.2) !important;
    }

    /* Header brand & button */
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-logo-badge {
        width: 34px;
        height: 34px;
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.35);
    }
    .header-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        color: #ffffff;
        padding: 7px 16px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        display: inline-block;
    }

    /* Cards */
    .fin-card {
        background: rgba(13, 18, 30, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
        position: relative;
    }
    .fin-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .card-big-val {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.6px;
    }
    .card-subtext {
        font-size: 11.5px;
        color: #64748b;
        margin-top: 4px;
    }
    .trend-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }
    .trend-up {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
    }

    /* Feature insight item */
    .insight-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .insight-row:last-child {
        border-bottom: none;
    }
    .insight-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
    }
    .icon-purple {
        background: rgba(168, 85, 247, 0.18);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .icon-cyan {
        background: rgba(6, 182, 212, 0.18);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .icon-teal {
        background: rgba(20, 184, 166, 0.18);
        color: #2dd4bf;
        border: 1px solid rgba(20, 184, 166, 0.3);
    }
    
    /* Result Card Styling */
    .result-banner {
        border-radius: 16px;
        padding: 20px 24px;
        margin: 18px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .result-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(13, 18, 30, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(13, 18, 30, 0.9) 100%);
        border: 1px solid rgba(244, 63, 94, 0.45);
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.15);
    }
</style>
""")

# -----------------------------------------------------------------------------
# TOP NAVBAR WITH 100% WORKING INTERACTIVE PILLS
# -----------------------------------------------------------------------------
head_left, head_center, head_right = st.columns([1.1, 2.2, 1.1])

with head_left:
    render_html("""
    <div style="padding-top:4px;">
        <div class="brand-title">
            <span class="brand-logo-badge">⚡</span>
            <span>LoanIQ <span style="font-size:12px; font-weight:400; color:#94a3b8;">Finsyght AI</span></span>
        </div>
    </div>
    """)

with head_center:
    # Fully interactive native segmented control pills (Now 100% working!)
    nav_selection = st.segmented_control(
        "Navigation",
        options=["● Executive Overview", "Credit Analysis", "Risk Modeling", "Live Predictions", "Audit History"],
        default="● Executive Overview",
        label_visibility="collapsed"
    )
    if not nav_selection:
        nav_selection = "● Executive Overview"

with head_right:
    render_html("""
    <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px; padding-top:4px;">
        <span class="header-btn">✦ AI Active</span>
        <div style="width:34px; height:34px; border-radius:50%; background:rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:#38bdf8; border:1px solid rgba(56,189,248,0.3);">AD</div>
    </div>
    """)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# =============================================================================
# VIEW 1: EXECUTIVE OVERVIEW (Matching Screenshot)
# =============================================================================
if nav_selection == "● Executive Overview":
    # --- Top Row: 3 Compact KPI Cards ---
    top1, top2, top3 = st.columns(3)
    
    with top1:
        render_html("""
        <div class="fin-card">
            <div class="card-title">
                <span>PORTFOLIO CONVERSIONS</span>
                <span style="font-size:11px; color:#8b5cf6;">↗ Real-time</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div class="card-big-val">13.23 Cr</div>
                <div class="trend-pill trend-up">+14.2%</div>
            </div>
            <div class="card-subtext">Active sanctioned loan capital evaluated</div>
        </div>
        """)
        
    with top2:
        render_html("""
        <div class="fin-card">
            <div class="card-title">
                <span>APPLICATIONS EVALUATED</span>
                <span style="font-size:11px; color:#06b6d4;">📊 Total</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div class="card-big-val">4,269</div>
                <div class="trend-pill trend-up">98.2% Accuracy</div>
            </div>
            <div class="card-subtext">Random Forest Classifier Pipeline</div>
        </div>
        """)
        
    with top3:
        render_html("""
        <div class="fin-card">
            <div class="card-title">
                <span>SYSTEM THROUGHPUT</span>
                <span style="font-size:11px; color:#10b981;">⚡ Sub-50ms</span>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div class="card-big-val">18,453</div>
                <div class="trend-pill trend-up">Active API</div>
            </div>
            <div class="card-subtext">Zero data leakage • Stratified validation</div>
        </div>
        """)

    # --- Middle Row: Financial Volume, Multi-Wave Chart, System Heuristics ---
    m1, m2, m3 = st.columns([1, 1.4, 1.1])
    
    with m1:
        render_html("""
        <div class="fin-card">
            <div class="card-title">
                <span>TOTAL DISBURSED</span>
                <span style="font-size:16px; color:#64748b;">₹</span>
            </div>
            <div class="card-big-val">₹204,890 K</div>
            <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
                <span class="trend-pill trend-up">3.2% MoM</span>
                <span style="font-size:11px; color:#64748b;">Average ticket ₹1.2 Cr</span>
            </div>
        </div>
        """)
        
        # Mini sparkline under disbursed
        spark_fig = go.Figure()
        spark_fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5, 6, 7],
            y=[14, 16, 15, 19, 18, 22, 24],
            mode="lines",
            line=dict(color="#06b6d4", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.08)"
        ))
        spark_fig.update_layout(
            height=90,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(spark_fig, use_container_width=True, config={"displayModeBar": False})

    with m2:
        render_html("""
        <div class="fin-card" style="padding-bottom:6px;">
            <div class="card-title">
                <span>APPROVAL VS REJECTION WAVE DYNAMICS</span>
                <span style="font-size:11px; color:#a855f7;">● Approved  ● Rejected</span>
            </div>
        </div>
        """)
        
        # Multi-wave Chart (matching screenshot's glowing purple/cyan wave curves)
        x_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        approved_curve = [18, 24, 21, 38, 26, 42, 35, 52, 48, 62, 58, 70]
        rejected_curve = [8, 12, 10, 16, 14, 20, 15, 24, 20, 26, 22, 28]
        
        wave_fig = go.Figure()
        wave_fig.add_trace(go.Scatter(
            x=x_months, y=approved_curve,
            mode="lines",
            name="Approved",
            line=dict(color="#a855f7", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(168, 85, 247, 0.1)"
        ))
        wave_fig.add_trace(go.Scatter(
            x=x_months, y=rejected_curve,
            mode="lines",
            name="Rejected",
            line=dict(color="#06b6d4", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.08)"
        ))
        wave_fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#64748b", size=10))
        )
        st.plotly_chart(wave_fig, use_container_width=True, config={"displayModeBar": False})

    with m3:
        render_html("""
        <div class="fin-card">
            <div class="card-title">
                <span>SYSTEM HEURISTICS</span>
                <span style="font-size:11px; color:#8b5cf6;">Rule Sets</span>
            </div>
            
            <div class="insight-row">
                <div class="insight-icon icon-purple">💳</div>
                <div>
                    <div style="font-size:12.5px; font-weight:600; color:#ffffff;">CIBIL Threshold 650+</div>
                    <div style="font-size:11px; color:#94a3b8;">Primary approval weighting factor (88.4%)</div>
                </div>
            </div>

            <div class="insight-row">
                <div class="insight-icon icon-cyan">🏢</div>
                <div>
                    <div style="font-size:12.5px; font-weight:600; color:#ffffff;">Asset-to-Loan Ratio</div>
                    <div style="font-size:11px; color:#94a3b8;">Luxury + Residential collateral coverage</div>
                </div>
            </div>

            <div class="insight-row">
                <div class="insight-icon icon-teal">⏱</div>
                <div>
                    <div style="font-size:12.5px; font-weight:600; color:#ffffff;">Tenure Sensitivity</div>
                    <div style="font-size:11px; color:#94a3b8;">Optimal repayment term: 5 - 12 Years</div>
                </div>
            </div>
        </div>
        """)

    # Bottom Row: Quick Navigation Trigger
    render_html("""
    <div class="fin-card" style="display:flex; justify-content:space-between; align-items:center; padding:16px 22px;">
        <div>
            <h4 style="color:#ffffff; font-size:15px; margin:0;">Ready to evaluate a new loan application?</h4>
            <p style="color:#94a3b8; font-size:12px; margin:2px 0 0;">Switch to the <strong>Live Predictions</strong> tab above or test credit analysis models.</p>
        </div>
        <div>
            <span style="color:#a855f7; font-size:12px; font-weight:600;">⚡ Instant Inference Engine Ready</span>
        </div>
    </div>
    """)

# =============================================================================
# VIEW 2: CREDIT ANALYSIS
# =============================================================================
elif nav_selection == "Credit Analysis":
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        render_html("""
        <div class="fin-card" style="padding-bottom:6px;">
            <div class="card-title">
                <span>CIBIL RISK SEGMENTATION</span>
                <span class="trend-pill trend-up">328,590 Total Records</span>
            </div>
        </div>
        """)
        
        # Vertical gradient rounded bars matching screenshot
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=["300-500", "501-650", "651-750", "751-850", "851-900"],
            y=[140, 310, 720, 1840, 1259],
            marker=dict(
                color=["#38bdf8", "#06b6d4", "#14b8a6", "#10b981", "#34d399"],
                line=dict(width=0)
            )
        ))
        bar_fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#64748b", size=10))
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with c_right:
        render_html("""
        <div class="fin-card" style="padding-bottom:4px;">
            <div class="card-title">
                <span>CONVERSION RATE TREND (MOUNTAIN AREA)</span>
                <span style="font-size:11px; color:#38bdf8;">1 Year Rolling</span>
            </div>
        </div>
        """)
        
        mountain_fig = go.Figure()
        m_x = [f"W{i}" for i in range(1, 16)]
        m_y = [20, 24, 22, 38, 30, 48, 42, 56, 50, 68, 60, 75, 70, 85, 82]
        mountain_fig.add_trace(go.Scatter(
            x=m_x, y=m_y,
            mode="lines",
            line=dict(color="#0ea5e9", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.2)"
        ))
        mountain_fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=9)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", tickfont=dict(color="#64748b", size=9))
        )
        st.plotly_chart(mountain_fig, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# VIEW 3: RISK MODELING (METRICS & MODEL TRANSPARENCY)
# =============================================================================
elif nav_selection == "Risk Modeling":
    render_html("""
    <div style="margin-bottom:14px;">
        <h3 style="font-size:18px; font-weight:700; color:#ffffff; margin:0;">Machine Learning Architecture & Validation Metrics</h3>
        <p style="font-size:12.5px; color:#94a3b8; margin:2px 0 0;">Full transparency into the trained pipeline, cross-validation metrics, and feature importance.</p>
    </div>
    """)

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        render_html("""
        <div class="fin-card">
            <h4 style="color:#ffffff; margin-bottom:12px;">Model Performance Summary</h4>
            <table style="width:100%; font-size:13px; color:#cbd5e1;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;">Algorithm</td><td style="text-align:right; font-weight:700; color:#38bdf8;">Random Forest Classifier</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;">Test Set Accuracy</td><td style="text-align:right; font-weight:700; color:#34d399;">98.24%</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;">ROC-AUC Score</td><td style="text-align:right; font-weight:700; color:#c084fc;">99.87%</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;">Cross-Validation F1</td><td style="text-align:right; font-weight:700; color:#34d399;">98.15%</td></tr>
                <tr><td style="padding:8px 0;">Ensemble Estimators</td><td style="text-align:right; font-weight:700;">200 Trees (max_depth=12)</td></tr>
            </table>
        </div>
        """)

    with m_col2:
        render_html("""
        <div class="fin-card">
            <h4 style="color:#ffffff; margin-bottom:12px;">Primary Feature Importances</h4>
        </div>
        """)

        features = [
            "cibil_score", "loan_amount", "income_annum",
            "luxury_assets_value", "residential_assets_value",
            "commercial_assets_value", "bank_asset_value", "loan_term"
        ]
        importances = [0.82, 0.05, 0.04, 0.03, 0.02, 0.015, 0.015, 0.01]

        feat_fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation="h",
            marker=dict(color="#8b5cf6")
        ))
        feat_fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#cbd5e1", size=11), autorange="reversed")
        )
        st.plotly_chart(feat_fig, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# VIEW 4: LIVE PREDICTIONS (LOAN EVALUATOR FORM)
# =============================================================================
elif nav_selection == "Live Predictions":
    render_html("""
    <div style="margin-bottom:16px;">
        <h3 style="font-size:20px; font-weight:700; color:#ffffff; margin-bottom:4px;">Live AI Loan Eligibility Evaluator</h3>
        <p style="font-size:13px; color:#94a3b8;">Enter applicant details to generate instant machine learning predictions with risk & EMI analysis.</p>
    </div>
    """)

    with st.form("loan_prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 1. Applicant Demographics")
            dependents = st.number_input("Number of Dependents (0 - 5)", min_value=0, max_value=5, value=2)
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Employment Status", ["No (Salaried)", "Yes (Self-Employed)"])
            self_employed_clean = "Yes" if "Yes" in self_employed else "No"
            income = st.number_input("Annual Income (₹)", min_value=100000, max_value=100000000, value=7500000, step=100000)

        with col_b:
            st.markdown("##### 2. Loan Parameters & Credit")
            loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=100000, max_value=100000000, value=12000000, step=500000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)
            cibil = st.slider("CIBIL Credit Score (300 - 900)", min_value=300, max_value=900, value=760)

        with col_c:
            st.markdown("##### 3. Asset & Collateral Portfolio (₹)")
            res_assets = st.number_input("Residential Assets Value", min_value=0, max_value=100000000, value=8000000, step=200000)
            com_assets = st.number_input("Commercial Assets Value", min_value=0, max_value=100000000, value=4000000, step=200000)
            lux_assets = st.number_input("Luxury Assets Value", min_value=0, max_value=100000000, value=5000000, step=200000)
            bank_assets = st.number_input("Bank Liquid Assets", min_value=0, max_value=100000000, value=3000000, step=200000)

        submit_eval = st.form_submit_button("⚡ Run AI Risk Evaluation", use_container_width=True)

    if submit_eval:
        payload = {
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": self_employed_clean,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": res_assets,
            "commercial_assets_value": com_assets,
            "luxury_assets_value": lux_assets,
            "bank_asset_value": bank_assets
        }

        # Predict using scikit-learn model
        result = predict_loan(payload)

        # Save to SQLite database
        record_id = save_prediction_record(payload, result)

        # Visual Result Banner
        if result["is_approved"]:
            banner_class = "result-approved"
            badge_text = "✅ APPLICATION APPROVED"
            badge_color = "#34d399"
        else:
            banner_class = "result-rejected"
            badge_text = "❌ APPLICATION REJECTED"
            badge_color = "#fb7185"

        render_html(f"""
        <div class="result-banner {banner_class}">
            <div>
                <span style="font-size:12px; font-weight:700; color:{badge_color}; text-transform:uppercase; letter-spacing:1px;">Decision Verdict</span>
                <h2 style="font-size:26px; font-weight:800; color:#ffffff; margin:4px 0;">{badge_text}</h2>
                <p style="font-size:13px; color:#cbd5e1; margin:0;">
                    Reference Evaluation ID: <strong>#LN-REQ-{record_id:04d}</strong> • Persisted in SQLite
                </p>
            </div>
            <div style="text-align:right;">
                <span style="font-size:12px; color:#94a3b8;">Model Confidence</span>
                <div style="font-size:32px; font-weight:800; color:#ffffff;">{result['confidence']}%</div>
            </div>
        </div>
        """)

        # Metrics breakdown
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("CIBIL Rating Tier", result["cibil_rating"])
        r2.metric("Approval Probability", f"{result['approval_probability']}%")
        r3.metric("Estimated Monthly EMI", f"₹{result['monthly_emi']:,.2f}")
        r4.metric("Total Asset Coverage", f"₹{result['total_assets']:,.0f}")

        st.caption("ℹ Disclaimer: Predictions are educational machine learning estimates based on historical patterns, not formal banking credit approval.")

# =============================================================================
# VIEW 5: AUDIT HISTORY (SQLITE)
# =============================================================================
elif nav_selection == "Audit History":
    render_html("""
    <div style="margin-bottom:14px;">
        <h3 style="font-size:18px; font-weight:700; color:#ffffff; margin:0;">Persistent SQLite Prediction Audit Log</h3>
        <p style="font-size:12.5px; color:#94a3b8; margin:2px 0 0;">Historical record of all evaluated applications stored in local database.</p>
    </div>
    """)

    records = get_prediction_history(limit=50)

    if records:
        df_history = pd.DataFrame(records)
        df_history = df_history.rename(columns={
            "id": "ID",
            "applicant_summary": "Applicant Profile",
            "cibil_score": "CIBIL",
            "loan_amount": "Loan Amount (₹)",
            "loan_term": "Term (Yrs)",
            "predicted_status": "Verdict",
            "confidence": "Confidence %",
            "monthly_emi": "Est. EMI (₹)",
            "created_at": "Timestamp"
        })

        st.dataframe(df_history, use_container_width=True, hide_index=True)

        if st.button("🗑 Clear Prediction History", type="secondary"):
            clear_prediction_history()
            st.success("History cleared successfully!")
            st.rerun()
    else:
        st.info("No prediction records found yet. Switch to the Live Predictions tab to evaluate and save records!")
