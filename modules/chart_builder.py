"""
Module: chart_builder.py
Description: Generates 7 interactive Plotly charts with enterprise styling for Thiranex Solutions.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import plotly.subplots as sp
from typing import Dict, Optional
from statsmodels.tsa.seasonal import seasonal_decompose


# Custom Enterprise Palette
PRIMARY_COLOR = "#6366F1"
SECONDARY_COLOR = "#8B5CF6"
ACCENT_COLOR = "#EC4899"
INFO_COLOR = "#06B6D4"
SUCCESS_COLOR = "#10B981"
DARK_BG = "#0F172A"
CARD_BG = "#1E293B"

COLOR_DISCRETE_SEQUENCE = [
    "#6366F1", "#8B5CF6", "#EC4899", "#06B6D4", "#10B981", 
    "#F59E0B", "#3B82F6", "#A855F7", "#F43F5E", "#14B8A6"
]


def _apply_theme(fig, title: str = ""):
    """Applies unified Plotly theme styling."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Outfit", size=18, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94A3B8"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1")
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    )
    return fig


def build_revenue_trend_chart(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Dual-axis chart: Bar chart for Revenue and Line chart for Order Count over time.
    """
    date_col = mappings.get("date")
    rev_col = mappings.get("revenue")

    if not date_col or date_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="Insufficient date data for trend analysis", showarrow=False, font=dict(size=14))
        return _apply_theme(fig, "Revenue & Order Volume Trend")

    # Aggregate by date (daily or weekly depending on span)
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    span_days = (temp_df[date_col].max() - temp_df[date_col].min()).days

    freq = "D" if span_days <= 60 else ("W" if span_days <= 365 else "M")
    agg_df = temp_df.groupby(pd.Grouper(key=date_col, freq=freq)).agg(
        Revenue=(rev_col, "sum") if rev_col else (date_col, "count"),
        Orders=(date_col, "count")
    ).reset_index()

    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    # Revenue Bar
    fig.add_trace(
        gg.Bar(
            x=agg_df[date_col],
            y=agg_df["Revenue"],
            name="Revenue ($)",
            marker_color=PRIMARY_COLOR,
            opacity=0.85,
            hovertemplate="<b>Date</b>: %{x|%b %d, %Y}<br><b>Revenue</b>: $%{y:,.2f}<extra></extra>"
        ),
        secondary_y=False
    )

    # Order Count Line
    fig.add_trace(
        gg.Scatter(
            x=agg_df[date_col],
            y=agg_df["Orders"],
            name="Order Count",
            mode="lines+markers",
            line=dict(color=ACCENT_COLOR, width=3),
            marker=dict(size=6, color="#FFFFFF", symbol="circle"),
            hovertemplate="<b>Date</b>: %{x|%b %d, %Y}<br><b>Orders</b>: %{y:,}<extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Order Count", secondary_y=True, showgrid=False)

    return _apply_theme(fig, "📈 Revenue Trend & Order Volume Over Time")


def build_monthly_heatmap(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Calendar Heatmap showing sales intensity by day-of-week vs. month.
    """
    date_col = mappings.get("date")
    rev_col = mappings.get("revenue")

    if not date_col or date_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="Insufficient date data for heatmap", showarrow=False)
        return _apply_theme(fig, "Sales Intensity Heatmap")

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df["Month"] = temp_df[date_col].dt.strftime("%b")
    temp_df["Month_Num"] = temp_df[date_col].dt.month
    temp_df["DayOfWeek"] = temp_df[date_col].dt.strftime("%a")
    temp_df["Day_Num"] = temp_df[date_col].dt.dayofweek

    pivot = temp_df.pivot_table(
        index="DayOfWeek",
        columns="Month",
        values=rev_col if rev_col else date_col,
        aggfunc="sum" if rev_col else "count"
    ).fillna(0)

    # Order days and months properly
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    pivot = pivot.reindex(index=[d for d in days_order if d in pivot.index],
                         columns=[m for m in months_order if m in pivot.columns])

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Day of Week", color="Sales ($)"),
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    return _apply_theme(fig, "🗓️ Sales Intensity Heatmap (Day vs. Month)")


def build_product_performance_matrix(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Scatter matrix plot (Price vs Quantity Sold) with bubble size proportional to Revenue.
    """
    prod_col = mappings.get("product")
    qty_col = mappings.get("quantity")
    price_col = mappings.get("price")
    rev_col = mappings.get("revenue")
    cat_col = mappings.get("category")

    if not prod_col or not qty_col or not price_col or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="Price, Quantity, and Product columns required", showarrow=False)
        return _apply_theme(fig, "Product Performance Matrix")

    agg_dict = {
        qty_col: "sum",
        price_col: "mean",
    }
    if rev_col:
        agg_dict[rev_col] = "sum"
    if cat_col:
        agg_dict[cat_col] = "first"

    perf_df = df.groupby(prod_col).agg(agg_dict).reset_index()

    if rev_col not in perf_df.columns:
        perf_df[rev_col] = perf_df[qty_col] * perf_df[price_col]

    fig = px.scatter(
        perf_df,
        x=price_col,
        y=qty_col,
        size=rev_col,
        color=cat_col if cat_col and cat_col in perf_df.columns else prod_col,
        hover_name=prod_col,
        size_max=45,
        color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
        labels={price_col: "Avg Unit Price ($)", qty_col: "Total Units Sold", rev_col: "Total Revenue ($)"}
    )

    return _apply_theme(fig, "🎯 Product Performance Matrix (Price vs. Quantity Sold)")


def build_category_donut_chart(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Donut chart of sales distribution by Category with drill-down ready styling.
    """
    cat_col = mappings.get("category") or mappings.get("product")
    rev_col = mappings.get("revenue")

    if not cat_col or cat_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="No Category data available", showarrow=False)
        return _apply_theme(fig, "Category Revenue Breakdown")

    cat_df = df.groupby(cat_col)[rev_col].sum().reset_index() if rev_col else df[cat_col].value_counts().reset_index()

    fig = px.pie(
        cat_df,
        names=cat_col,
        values=rev_col if rev_col else "count",
        hole=0.55,
        color_discrete_sequence=COLOR_DISCRETE_SEQUENCE
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#0F172A", width=2))
    )

    return _apply_theme(fig, "🍩 Revenue Distribution by Category")


def build_geographic_map(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Regional sales choropleth / bar map.
    """
    region_col = mappings.get("region")
    rev_col = mappings.get("revenue")

    if not region_col or region_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="No Region/Location column detected", showarrow=False)
        return _apply_theme(fig, "Geographic Sales Distribution")

    reg_df = df.groupby(region_col)[rev_col].sum().reset_index() if rev_col else df[region_col].value_counts().reset_index()

    fig = px.bar(
        reg_df,
        x=region_col,
        y=rev_col if rev_col else "count",
        color=rev_col if rev_col else "count",
        color_continuous_scale="Purples",
        labels={region_col: "Region", rev_col: "Total Revenue ($)"}
    )

    return _apply_theme(fig, "🌍 Geographic & Regional Sales Overview")


def build_salesperson_leaderboard(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Horizontal bar chart ranking sales reps by total revenue generated.
    """
    rep_col = mappings.get("salesperson")
    rev_col = mappings.get("revenue")

    if not rep_col or rep_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="No Salesperson column detected", showarrow=False)
        return _apply_theme(fig, "Salesperson Leaderboard")

    rep_df = df.groupby(rep_col)[rev_col].sum().sort_values(ascending=True).reset_index() if rev_col else df[rep_col].value_counts().sort_values(ascending=True).reset_index()

    fig = px.bar(
        rep_df,
        y=rep_col,
        x=rev_col if rev_col else "count",
        orientation="h",
        color=rev_col if rev_col else "count",
        color_continuous_scale="Plasma",
        labels={rep_col: "Sales Representative", rev_col: "Total Revenue ($)"}
    )

    return _apply_theme(fig, "🏆 Sales Representative Leaderboard")


def build_time_series_decomposition(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> gg.Figure:
    """
    Decomposes time series into Trend, Seasonality, and Residual components.
    """
    date_col = mappings.get("date")
    rev_col = mappings.get("revenue")

    if not date_col or date_col not in df.columns or df.empty:
        fig = gg.Figure()
        fig.add_annotation(text="Insufficient date data for time series decomposition", showarrow=False)
        return _apply_theme(fig, "Time Series Decomposition")

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    daily = temp_df.groupby(pd.Grouper(key=date_col, freq="D"))[rev_col].sum().fillna(0)

    if len(daily) < 14:
        fig = gg.Figure()
        fig.add_annotation(text="Requires at least 14 days of data for decomposition", showarrow=False)
        return _apply_theme(fig, "Time Series Decomposition")

    try:
        # Perform seasonal decomposition with period=7 (weekly seasonality)
        decomp = seasonal_decompose(daily, model="additive", period=7)
        
        fig = sp.make_subplots(rows=3, cols=1, subplot_titles=["Trend Component", "Seasonal Pattern", "Residual Noise"])

        fig.add_trace(gg.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend", line=dict(color=PRIMARY_COLOR, width=2)), row=1, col=1)
        fig.add_trace(gg.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal", line=dict(color=ACCENT_COLOR, width=2)), row=2, col=1)
        fig.add_trace(gg.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual", mode="markers", marker=dict(color=INFO_COLOR, size=4)), row=3, col=1)

        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        return _apply_theme(fig, "📊 Time Series Decomposition (Trend, Seasonality, Residuals)")
    except Exception:
        # Fallback to simple moving average if statsmodels fails
        fig = gg.Figure()
        ma = daily.rolling(window=7).mean()
        fig.add_trace(gg.Scatter(x=daily.index, y=daily, name="Actual Revenue", line=dict(color="rgba(99,102,241,0.4)")))
        fig.add_trace(gg.Scatter(x=ma.index, y=ma, name="7-Day Moving Avg Trend", line=dict(color=ACCENT_COLOR, width=3)))
        return _apply_theme(fig, "📊 7-Day Trend Analysis (Moving Average)")
