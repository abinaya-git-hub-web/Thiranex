"""
Module: ai_insights.py
Description: Machine Learning and AI engine for Isolation Forest anomaly detection, 
             time series 30-day forecasting, smart business recommendations, 
             and automated natural language executive summary generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def detect_sales_anomalies(df: pd.DataFrame, mappings: Dict[str, Optional[str]], contamination: float = 0.03) -> pd.DataFrame:
    """
    Uses Scikit-Learn Isolation Forest algorithm to detect unusual revenue/quantity transactions.

    Args:
        df (pd.DataFrame): Filtered sales dataset.
        mappings (Dict[str, Optional[str]]): Column role mappings.
        contamination (float): Expected proportion of outliers in the data. Default 0.03.

    Returns:
        pd.DataFrame: Subset of df containing flagged anomalies with anomaly scores.
    """
    rev_col = mappings.get("revenue")
    qty_col = mappings.get("quantity")

    if df.empty or not rev_col or rev_col not in df.columns or len(df) < 10:
        return pd.DataFrame()

    features = [rev_col]
    if qty_col and qty_col in df.columns:
        features.append(qty_col)

    X = df[features].fillna(0)

    iso = IsolationForest(contamination=contamination, random_state=42)
    df_copy = df.copy()
    df_copy["Anomaly_Flag"] = iso.fit_predict(X)
    df_copy["Anomaly_Score"] = iso.decision_function(X)

    # Anomaly_Flag is -1 for outliers, 1 for inliers
    anomalies = df_copy[df_copy["Anomaly_Flag"] == -1].sort_values("Anomaly_Score", ascending=True)
    return anomalies


def generate_sales_forecast(df: pd.DataFrame, mappings: Dict[str, Optional[str]], forecast_days: int = 30) -> pd.DataFrame:
    """
    Builds a 30-day time series forecast using Holt-Winters Exponential Smoothing (or polynomial fallback).

    Args:
        df (pd.DataFrame): Filtered sales dataset.
        mappings (Dict[str, Optional[str]]): Column role mappings.
        forecast_days (int): Number of future days to project. Default 30.

    Returns:
        pd.DataFrame: Projected sales DataFrame with Date, Predicted_Revenue, Lower_Bound, Upper_Bound.
    """
    date_col = mappings.get("date")
    rev_col = mappings.get("revenue")

    if not date_col or date_col not in df.columns or not rev_col or rev_col not in df.columns or df.empty:
        return pd.DataFrame()

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    daily = temp_df.groupby(pd.Grouper(key=date_col, freq="D"))[rev_col].sum().fillna(0)

    if len(daily) < 10:
        return pd.DataFrame()

    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    try:
        # Fit Holt-Winters Exponential Smoothing
        model = ExponentialSmoothing(daily, trend="add", seasonal=None, initialization_method="estimated")
        fit_model = model.fit()
        forecast_vals = fit_model.forecast(forecast_days)
        
        # Calculate standard deviation of residuals for confidence bands
        residuals = daily - fit_model.fittedvalues
        std_err = np.std(residuals)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Revenue": np.maximum(0, forecast_vals.values),
            "Lower_Bound": np.maximum(0, forecast_vals.values - (1.96 * std_err)),
            "Upper_Bound": forecast_vals.values + (1.96 * std_err)
        })

    except Exception:
        # Graceful fallback: Polynomial trend extrapolation
        x = np.arange(len(daily))
        y = daily.values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        future_x = np.arange(len(daily), len(daily) + forecast_days)
        future_y = np.maximum(0, p(future_x))

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Revenue": future_y,
            "Lower_Bound": np.maximum(0, future_y * 0.85),
            "Upper_Bound": future_y * 1.15
        })

    return forecast_df


def generate_smart_recommendations(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> List[Dict[str, str]]:
    """
    Analyzes sales dataset patterns to formulate actionable strategic business recommendations.

    Args:
        df (pd.DataFrame): Cleaned sales DataFrame.
        mappings (Dict[str, Optional[str]]): Column role mappings.

    Returns:
        List[Dict[str, str]]: List of recommendation cards with Title, Icon, Insight, and Recommendation.
    """
    recs = []

    prod_col = mappings.get("product")
    rev_col = mappings.get("revenue")
    date_col = mappings.get("date")
    rep_col = mappings.get("salesperson")

    if df.empty:
        return recs

    # Recommendation 1: Underperforming Products
    if prod_col and rev_col and prod_col in df.columns:
        prod_rev = df.groupby(prod_col)[rev_col].sum().sort_values(ascending=True)
        if len(prod_rev) >= 3:
            bottom_3 = list(prod_rev.head(3).index)
            recs.append({
                "title": "⚠️ Underperforming Product Strategy",
                "category": "Portfolio Optimization",
                "icon": "📉",
                "insight": f"Lowest performing offerings: **{', '.join(bottom_3)}** generated under ${prod_rev.iloc[0]:,.2f} total revenue.",
                "action": "Consider bundling these underperformers with top-selling suites, re-evaluating price positioning, or reallocating marketing budget."
            })

    # Recommendation 2: Best Time for Promotions
    if date_col and date_col in df.columns:
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df["DayOfWeek"] = temp_df[date_col].dt.strftime("%A")
        
        day_sales = temp_df.groupby("DayOfWeek")[rev_col].sum() if rev_col else temp_df["DayOfWeek"].value_counts()
        best_day = day_sales.idxmax()
        worst_day = day_sales.idxmin()

        recs.append({
            "title": "📅 Promotional Timing Optimization",
            "category": "Marketing Campaign",
            "icon": "⚡",
            "insight": f"Sales intensity peaks significantly on **{best_day}s**, whereas **{worst_day}s** witness the lowest customer conversion velocity.",
            "action": f"Schedule product launches and flash discount campaigns on **{best_day}s** to capitalize on peak buying momentum."
        })

    # Recommendation 3: Salesperson Risk / Concentration Analysis
    if rep_col and rev_col and rep_col in df.columns:
        rep_sales = df.groupby(rep_col)[rev_col].sum()
        total_rev = rep_sales.sum()
        top_rep = rep_sales.idxmax()
        top_rep_pct = (rep_sales.max() / total_rev) * 100 if total_rev > 0 else 0

        if top_rep_pct > 25:
            recs.append({
                "title": "🎯 Revenue Concentration Risk",
                "category": "Sales Operations",
                "icon": "👥",
                "insight": f"Top sales representative **{top_rep}** accounts for **{top_rep_pct:.1f}%** of total enterprise revenue.",
                "action": "Implement knowledge-sharing workshops and distribute high-value lead channels to balance target achievement across team members."
            })

    return recs


def generate_executive_summary(kpis: Dict[str, Any], anomalies_count: int, forecast_df: pd.DataFrame) -> str:
    """
    Generates a natural language executive summary report for executive leadership.

    Args:
        kpis (Dict[str, Any]): Calculated KPI metrics.
        anomalies_count (int): Total detected anomaly transactions.
        forecast_df (pd.DataFrame): 30-day forecasted data.

    Returns:
        str: Markdown-formatted executive summary statement.
    """
    rev = kpis.get("total_revenue", 0.0)
    rev_delta = kpis.get("revenue_delta", 0.0)
    orders = kpis.get("total_orders", 0)
    aov = kpis.get("aov", 0.0)
    top_prod = kpis.get("top_product", "N/A")

    forecast_total = forecast_df["Predicted_Revenue"].sum() if not forecast_df.empty else 0.0
    growth_direction = "positive growth" if rev_delta >= 0 else "contraction"

    summary = f"""
### 📊 Executive Summary for Thiranex Solutions Leadership

During the selected evaluation window, Thiranex Solutions generated **${rev:,.2f}** in net revenue across **{orders:,}** total client orders, achieving an Average Order Value (AOV) of **${aov:,.2f}**.

**Key Strategic Insights:**
- **Revenue Momentum:** Sales demonstrated a **{abs(rev_delta):.1f}% {growth_direction}** compared to the preceding comparison period.
- **Star Performer:** The leading product driving revenue growth was **{top_prod}**, contributing significantly to top-line performance.
- **AI Anomaly Detection:** Isolation Forest algorithm identified **{anomalies_count} anomalous transaction(s)** requiring audit or review for potential volume surges or outliers.
- **30-Day Outlook:** Predictive modeling projects an estimated **${forecast_total:,.2f}** in cumulative sales over the upcoming 30-day horizon.

*Recommendation:* Maintain focus on high-velocity enterprise software solutions while addressing identified sales distribution bottlenecks.
"""
    return summary.strip()
