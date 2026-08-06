"""
=============================================================================
Thiranex Solutions — UI Components & Custom Dark Glassmorphic Theme Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def inject_custom_css():
    """
    Injects enterprise glassmorphic dark theme styling, animated glow effects,
    and responsive layout CSS rules.
    """
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #0F172A 50%, #1E1035 100%);
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139, 92, 246, 0.4);
    }

    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.2rem;
    }

    .metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
        border: none;
        padding: 0 16px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }

    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35);
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header_banner():
    """
    Renders top glassmorphic header banner with Thiranex Solutions branding.
    """
    banner_html = """
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9)); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 2.2rem;">⚡</span>
                    <div>
                        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(135deg, #A855F7 0%, #3B82F6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            THIRANEX SOLUTIONS
                        </h1>
                        <p style="margin: 0.2rem 0 0 0; color: #94A3B8; font-size: 0.92rem; font-weight: 500;">
                            Enterprise Predictive Analytics & AutoML Forecasting Engine v3.5
                        </p>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; margin-top: 0.5rem;">
                <span style="background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.5); color: #C084FC; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                    ● AutoML Active
                </span>
                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); color: #34D399; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                    ● 13 Plotly Visualizations
                </span>
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.5); color: #60A5FA; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                    ● 10 Novel Features
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, icon: str = "📊", delta: str = None, delta_color: str = "#10B981"):
    """
    Renders a glassmorphic KPI summary card.
    """
    delta_html = f'<div class="metric-delta" style="color: {delta_color};">{delta}</div>' if delta else ''
    card_html = f"""
    <div class="glass-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_onboarding_tour():
    """
    Renders expandable step-by-step user workflow and business-ready outputs guide.
    """
    with st.expander("🗺️ End-to-End User Workflow & Business-Ready Outputs Guide", expanded=False):
        st.markdown(
            """
            ### 📌 7-Step Operational Workflow
            1. **Step 1 — Data Import**: Upload CSV, Excel, JSON file or use 3+ Year Synthetic Demo Data.
            2. **Step 2 — Data Preprocessing**: Automated cleaning (imputation, IQR/Isolation Forest outlier removal, ADF/KPSS stationarity tests).
            3. **Step 3 — Model Selection**: Choose a dedicated algorithm (SARIMA, Prophet, Holt-Winters, Ridge, Random Forest, XGBoost, SVR, Neural Net) or use **AutoML Mode**.
            4. **Step 4 — Set Forecast Parameters**: Set forecast horizon (7-365 days), confidence level (80-99%), and lag feature depth.
            5. **Step 5 — Model Training & Diagnostics**: System trains model and generates out-of-sample performance metrics (RMSE, MAE, MAPE, $R^2$), residual diagnostics, and 13 interactive Plotly visualizations.
            6. **Step 6 — What-If Scenario Planning**: Explore interactive sliders for marketing spend shifts, pricing changes, and competitor shifts.
            7. **Step 7 — Export & Share Reports**: Download prediction CSVs, ReportLab PDF Executive Briefings, or programmatically query the REST API endpoint.

            ---

            ### 🎯 7 Business-Ready Outputs Included
            - 📊 **Executive Dashboard**: Leadership KPI metrics & high-level trend projections.
            - 📈 **Detailed Forecast Report**: Overlay plots with 95% confidence intervals and future horizon curves.
            - 🏆 **Model Performance Report**: Side-by-side leaderboard ranking all 8 model architectures.
            - 🛡️ **Risk Assessment**: Prediction uncertainty spread, P10/P50/P90 interval totals, and strategic risk ratings.
            - 🔮 **Scenario Planning**: Interactive What-if simulation comparison charts & net financial delta.
            - 💡 **AI Executive Summary**: Natural language plain-English briefing summarizing trends, performance, and risks.
            - 📑 **Recommendation Report**: Actionable data-driven business guidance & PDF briefing export.
            """
        )


def render_help_sidebar():
    """
    Renders help sidebar documentation & FAQs.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ❓ Help & FAQs")
    with st.sidebar.expander("What models are supported?"):
        st.markdown(
            "- **Time-Series**: SARIMA, Facebook Prophet, Holt-Winters, VAR\n"
            "- **Regression**: Ridge, Lasso, ElasticNet, Random Forest, XGBoost, SVR, Neural Network (MLP/LSTM)"
        )
    with st.sidebar.expander("How does AutoML work?"):
        st.markdown("AutoML fits all 8 models automatically, ranks them by RMSE, and creates a top-3 weighted ensemble.")


def render_footer():
    """
    Renders footer notice.
    """
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding-bottom: 2rem;">
            © 2026 <b>Thiranex Solutions</b> — Enterprise Predictive Analytics Suite | Built with Streamlit, Scikit-learn, Statsmodels & Plotly ❤️
        </div>
        """,
        unsafe_allow_html=True
    )
