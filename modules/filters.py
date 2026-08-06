"""
=============================================================================
Thiranex Solutions — Sidebar Controls & Dynamic Filters Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple


def render_sidebar_controls(df_raw: pd.DataFrame, detected_mappings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders interactive sidebar controls for modeling parameters, algorithms,
    data cleaning settings, scenario variables, and confidence bounds.
    """
    st.sidebar.markdown("### ⚙️ Predictive Model Controls")

    # 1. Target & Date Column Selection
    date_cols = df_raw.columns.tolist()
    target_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]

    date_col = st.sidebar.selectbox(
        "Date / Time Column",
        options=date_cols,
        index=date_cols.index(detected_mappings["date_col"]) if detected_mappings["date_col"] in date_cols else 0
    )

    target_col = st.sidebar.selectbox(
        "Target Variable Column",
        options=target_cols if target_cols else date_cols,
        index=target_cols.index(detected_mappings["target_col"]) if detected_mappings["target_col"] in target_cols else 0
    )

    # 2. Model Selection
    model_options = [
        "🤖 AutoML (Auto-Select Best Model)",
        "✨ Top-3 Weighted Ensemble",
        "SARIMA / SARIMAX",
        "Prophet (Facebook)",
        "Holt-Winters Exponential Smoothing",
        "Ridge Regression",
        "Random Forest Regressor",
        "XGBoost Regressor",
        "Support Vector Regressor (SVR)",
        "Neural Network (LSTM / MLP)"
    ]

    selected_model = st.sidebar.selectbox(
        "Forecast Model Algorithm",
        options=model_options,
        index=0,
        help="AutoML automatically tests all models and ranks them by RMSE."
    )

    # 3. Forecast Horizon & Train-Test Split
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Horizon & Split Ratios")

    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Periods/Days)",
        min_value=7,
        max_value=365,
        value=30,
        step=1
    )

    test_split_ratio = st.sidebar.slider(
        "Train-Test Split Ratio (Test %)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of historical time series reserved for out-of-sample testing."
    ) / 100.0

    confidence_level = st.sidebar.selectbox(
        "Prediction Confidence Level",
        options=[0.80, 0.90, 0.95, 0.99],
        index=2,
        format_func=lambda x: f"{int(x*100)}% Confidence Band"
    )

    # 4. Data Cleaning & Feature Toggles
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Cleaning & Feature Engineering")

    impute_method = st.sidebar.selectbox(
        "Missing Value Imputation",
        options=["Linear Interpolation", "Forward Fill (ffill)", "Backward Fill (bfill)", "Mean Imputation", "Median Imputation"],
        index=0
    )

    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method",
        options=["IQR Method", "Z-Score Method", "Isolation Forest", "None"],
        index=0
    )

    smooth_method = st.sidebar.selectbox(
        "Data Smoothing",
        options=["None", "Moving Average", "Exponential Smoothing"],
        index=0
    )

    max_lags = st.sidebar.slider("Lag Features Count", min_value=1, max_value=14, value=7)
    include_calendar = st.sidebar.checkbox("Include Date/Time Calendar Features", value=True)
    include_fourier = st.sidebar.checkbox("Include Fourier Seasonal Terms", value=True)

    # 5. What-If Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ What-If Scenario Sliders")

    marketing_change_pct = st.sidebar.slider("Marketing Spend Shift (%)", -50, 50, 0, step=5)
    price_change_pct = st.sidebar.slider("Product Pricing Shift (%)", -30, 30, 0, step=5)
    competitor_impact_pct = st.sidebar.slider("Competitor Activity Shift (%)", -40, 40, 0, step=5)

    budget_target = st.sidebar.number_input("Budget Target Amount ($)", value=25000.0, step=1000.0)

    return {
        "date_col": date_col,
        "target_col": target_col,
        "selected_model": selected_model,
        "forecast_horizon": forecast_horizon,
        "test_split_ratio": test_split_ratio,
        "confidence_level": confidence_level,
        "impute_method": impute_method,
        "outlier_method": outlier_method,
        "smooth_method": smooth_method,
        "max_lags": max_lags,
        "include_calendar": include_calendar,
        "include_fourier": include_fourier,
        "marketing_change_pct": marketing_change_pct,
        "price_change_pct": price_change_pct,
        "competitor_impact_pct": competitor_impact_pct,
        "budget_target": budget_target
    }
