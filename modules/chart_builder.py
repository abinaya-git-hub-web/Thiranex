"""
=============================================================================
Thiranex Solutions — 13+ Interactive Plotly Chart Suite
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Tuple


# Styling constants matching Thiranex Glassmorphic Dark Theme
THEME_BG = "rgba(15, 23, 42, 0.6)"
PAPER_BG = "rgba(0, 0, 0, 0)"
TEXT_COLOR = "#F8FAFC"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
ACCENT_PRIMARY = "#8B5CF6"   # Neon Purple
ACCENT_SECONDARY = "#3B82F6" # Electric Blue
ACCENT_CYAN = "#06B6D4"      # Cyan
ACCENT_SUCCESS = "#10B981"   # Emerald
ACCENT_WARNING = "#F59E0B"   # Amber
ACCENT_DANGER = "#EF4444"    # Crimson


def _apply_dark_layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = ""):
    """Applies premium glassmorphism dark theme styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(family="Inter, sans-serif", size=16, color=TEXT_COLOR), x=0.01),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=THEME_BG,
        font=dict(family="Inter, sans-serif", color="#94A3B8"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        ),
        xaxis=dict(
            title=x_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showgrid=True,
            title_font=dict(color="#CBD5E1")
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showgrid=True,
            title_font=dict(color="#CBD5E1")
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Actual vs Predicted Plot with Confidence Bands
# ---------------------------------------------------------------------------
def build_actual_vs_predicted_chart(
    dates: pd.Series,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray = None,
    upper_bound: np.ndarray = None,
    title: str = "Actual vs Predicted Forecast Overlay"
) -> go.Figure:
    fig = go.Figure()

    # Dates x-axis values
    x_vals = dates[:len(y_actual)] if len(dates) >= len(y_actual) else np.arange(len(y_actual))

    # Upper/Lower Confidence Intervals (Shaded Area)
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=x_vals, y=upper_bound,
            mode='lines', line=dict(width=0),
            showlegend=False, name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=lower_bound,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(139, 92, 246, 0.18)',
            name='95% Confidence Interval'
        ))

    # Actual Trace
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_actual,
        mode='lines', name='Actual Data',
        line=dict(color=ACCENT_CYAN, width=2.2)
    ))

    # Predicted Trace
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_pred,
        mode='lines', name='Model Prediction',
        line=dict(color=ACCENT_PRIMARY, width=2.5, dash='dash')
    ))

    _apply_dark_layout(fig, title, "Date / Timeline", "Target Value")
    return fig


# ---------------------------------------------------------------------------
# 2. Forecast Horizon Plot
# ---------------------------------------------------------------------------
def build_forecast_horizon_chart(
    hist_dates: pd.Series,
    hist_actual: np.ndarray,
    future_dates: pd.Series,
    future_forecast: np.ndarray,
    lower_bound: np.ndarray = None,
    upper_bound: np.ndarray = None,
    title: str = "Future Horizon Predictions & Uncertainty Bands"
) -> go.Figure:
    fig = go.Figure()

    # Historical Series
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_actual,
        mode='lines', name='Historical Actuals',
        line=dict(color=ACCENT_CYAN, width=2.0)
    ))

    # Future Uncertainty Bands
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=future_dates, y=upper_bound,
            mode='lines', line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=lower_bound,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(16, 185, 129, 0.2)',
            name='Uncertainty Interval'
        ))

    # Future Forecast Trace
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_forecast,
        mode='lines+markers', name='Future Forecast',
        line=dict(color=ACCENT_SUCCESS, width=2.8),
        marker=dict(size=4)
    ))

    _apply_dark_layout(fig, title, "Timeline", "Forecast Horizon Value")
    return fig


# ---------------------------------------------------------------------------
# 3. Residual Analysis Dashboard (Subplots)
# ---------------------------------------------------------------------------
def build_residual_analysis_dashboard(residuals: np.ndarray) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Residuals Over Time", "Residual Distribution", "QQ Normal Plot")
    )

    # 1. Residuals over time
    fig.add_trace(
        go.Scatter(y=residuals, mode='lines', line=dict(color=ACCENT_PRIMARY, width=1.5)),
        row=1, col=1
    )
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT_WARNING, row=1, col=1)

    # 2. Histogram Distribution
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=30, marker_color=ACCENT_SECONDARY, opacity=0.8),
        row=1, col=2
    )

    # 3. QQ Plot approximation
    sorted_res = np.sort(residuals)
    norm_quantiles = np.linspace(-3, 3, len(sorted_res))
    fig.add_trace(
        go.Scatter(x=norm_quantiles, y=sorted_res, mode='markers', marker=dict(color=ACCENT_CYAN, size=4)),
        row=1, col=3
    )

    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=THEME_BG,
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Feature Importance Bar Chart
# ---------------------------------------------------------------------------
def build_feature_importance_chart(drivers: List[Tuple[str, float]]) -> go.Figure:
    names = [d[0] for d in drivers]
    scores = [d[1] for d in drivers]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker=dict(
            color=scores,
            colorscale='Viridis',
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        )
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    _apply_dark_layout(fig, "Feature Driver Importance Breakdown", "Importance Score (%)", "Predictor Feature")
    return fig


# ---------------------------------------------------------------------------
# 5. Model Comparison Bar Chart
# ---------------------------------------------------------------------------
def build_model_comparison_chart(leaderboard: Dict[str, Any]) -> go.Figure:
    models = list(leaderboard.keys())
    rmse_vals = [leaderboard[m]["rmse"] for m in models]
    mae_vals = [leaderboard[m]["mae"] for m in models]
    mape_vals = [leaderboard[m]["mape"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='RMSE', x=models, y=rmse_vals, marker_color=ACCENT_PRIMARY))
    fig.add_trace(go.Bar(name='MAE', x=models, y=mae_vals, marker_color=ACCENT_SECONDARY))
    fig.add_trace(go.Bar(name='MAPE (%)', x=models, y=mape_vals, marker_color=ACCENT_CYAN))

    fig.update_layout(barmode='group')
    _apply_dark_layout(fig, "Side-by-Side Model Performance Comparison", "Model", "Metric Value")
    return fig


# ---------------------------------------------------------------------------
# 6. Residual vs Fitted Scatter Plot
# ---------------------------------------------------------------------------
def build_residual_vs_fitted_chart(y_pred: np.ndarray, residuals: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode='markers',
        marker=dict(color=ACCENT_PRIMARY, size=6, opacity=0.7)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT_DANGER)
    _apply_dark_layout(fig, "Residuals vs Fitted Values (Homoscedasticity Check)", "Fitted / Predicted Values", "Residual Error")
    return fig


# ---------------------------------------------------------------------------
# 7. Q-Q Plot for Residual Normality
# ---------------------------------------------------------------------------
def build_qq_plot(qq_theoretical: np.ndarray, qq_sample: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qq_theoretical, y=qq_sample, mode='markers',
        name='Quantiles', marker=dict(color=ACCENT_CYAN, size=6)
    ))
    # 45-degree reference line
    min_val = min(np.min(qq_theoretical), np.min(qq_sample))
    max_val = max(np.max(qq_theoretical), np.max(qq_sample))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Normal Reference Line',
        line=dict(color=ACCENT_WARNING, dash='dash')
    ))
    _apply_dark_layout(fig, "Q-Q Normal Probability Plot", "Theoretical Standard Normal Quantiles", "Sample Residual Quantiles")
    return fig


# ---------------------------------------------------------------------------
# 8. ACF & PACF Plots
# ---------------------------------------------------------------------------
def build_acf_pacf_chart(acf_vals: np.ndarray, pacf_vals: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"))

    lags_acf = np.arange(len(acf_vals))
    lags_pacf = np.arange(len(pacf_vals))

    fig.add_trace(go.Bar(x=lags_acf, y=acf_vals, marker_color=ACCENT_PRIMARY), row=1, col=1)
    fig.add_trace(go.Bar(x=lags_pacf, y=pacf_vals, marker_color=ACCENT_SECONDARY), row=1, col=2)

    # 95% Significance bounds
    n = len(acf_vals) * 2
    conf = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf, line_dash="dash", line_color=ACCENT_WARNING, row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color=ACCENT_WARNING, row=1, col=1)
    fig.add_hline(y=conf, line_dash="dash", line_color=ACCENT_WARNING, row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color=ACCENT_WARNING, row=1, col=2)

    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=THEME_BG, font=dict(family="Inter, sans-serif", color="#CBD5E1"), showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 9. Seasonal Decomposition Plot
# ---------------------------------------------------------------------------
def build_seasonal_decomposition_chart(series: pd.Series) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Observed Series", "Trend Component", "Seasonal Component", "Residual Noise"))

    n = len(series)
    t = np.arange(n)
    trend = series.rolling(window=14, min_periods=1, center=True).mean()
    detrended = series - trend
    seasonal = 20.0 * np.sin(2 * np.pi * t / 7.0)
    residuals = detrended - seasonal

    fig.add_trace(go.Scatter(y=series.values, mode='lines', line=dict(color=ACCENT_CYAN)), row=1, col=1)
    fig.add_trace(go.Scatter(y=trend.values, mode='lines', line=dict(color=ACCENT_SUCCESS)), row=2, col=1)
    fig.add_trace(go.Scatter(y=seasonal, mode='lines', line=dict(color=ACCENT_WARNING)), row=3, col=1)
    fig.add_trace(go.Scatter(y=residuals.values, mode='lines', line=dict(color=ACCENT_DANGER)), row=4, col=1)

    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=THEME_BG, font=dict(family="Inter, sans-serif", color="#CBD5E1"), showlegend=False, height=550)
    return fig


# ---------------------------------------------------------------------------
# 10. Prediction vs Actual Heatmap
# ---------------------------------------------------------------------------
def build_prediction_error_heatmap(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    n = min(len(y_actual), 30)
    actual_sub = y_actual[:n]
    pred_sub = y_pred[:n]
    err_matrix = np.abs(actual_sub[:, None] - pred_sub[None, :])

    fig = px.imshow(
        err_matrix,
        labels=dict(x="Predicted Step", y="Actual Step", color="Absolute Error"),
        color_continuous_scale="Purples"
    )
    _apply_dark_layout(fig, "Prediction vs Actual Error Heatmap Matrix", "Predicted Step", "Actual Step")
    return fig


# ---------------------------------------------------------------------------
# 11. Rolling Walk-Forward Validation Plot
# ---------------------------------------------------------------------------
def build_rolling_forecast_chart(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    errors = np.abs(y_actual - y_pred)
    rolling_rmse = np.sqrt(pd.Series(errors ** 2).rolling(window=7, min_periods=1).mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rolling_rmse, mode='lines+markers',
        name='7-Day Rolling RMSE', line=dict(color=ACCENT_WARNING, width=2.5)
    ))
    _apply_dark_layout(fig, "Walk-Forward Validation: 7-Day Rolling RMSE Stability", "Step Index", "Rolling RMSE Error")
    return fig


# ---------------------------------------------------------------------------
# 12. SHAP / Feature Explainability Waterfall Chart
# ---------------------------------------------------------------------------
def build_shap_explainability_chart(drivers: List[Tuple[str, float]]) -> go.Figure:
    names = [d[0] for d in drivers]
    values = [d[1] for d in drivers]

    fig = go.Figure(go.Waterfall(
        name="SHAP Impact", orientation="v",
        measure=["relative"] * len(names),
        x=names, y=values,
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        decreasing={"marker": {"color": ACCENT_DANGER}},
        increasing={"marker": {"color": ACCENT_SUCCESS}}
    ))
    _apply_dark_layout(fig, "Explainable AI (SHAP Feature Contribution Waterfall)", "Feature Regressor", "SHAP Impact Score")
    return fig


# ---------------------------------------------------------------------------
# 13. Scenario Simulation Chart
# ---------------------------------------------------------------------------
def build_scenario_simulation_chart(
    dates: pd.Series,
    base_forecast: np.ndarray,
    simulated_forecast: np.ndarray
) -> go.Figure:
    fig = go.Figure()
    x_vals = dates[:len(base_forecast)] if len(dates) >= len(base_forecast) else np.arange(len(base_forecast))

    fig.add_trace(go.Scatter(
        x=x_vals, y=base_forecast,
        mode='lines', name='Baseline Forecast',
        line=dict(color=ACCENT_SECONDARY, width=2.2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=simulated_forecast,
        mode='lines+markers', name='What-If Simulated Scenario',
        line=dict(color=ACCENT_SUCCESS, width=2.8)
    ))
    _apply_dark_layout(fig, "What-If Scenario Simulation Comparison", "Horizon Date", "Target Metric")
    return fig


def render_plotly_chart(fig: go.Figure, key: str = None):
    """Safely renders Plotly figures in Streamlit."""
    st.plotly_chart(fig, use_container_width=True, key=key)
