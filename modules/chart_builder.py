"""
=============================================================================
Thiranex Solutions — Interactive Plotly Visualizations Engine (12+ Charts)
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
from typing import Dict, Any, List

# Plotly Theme Colors
DARK_TEMPLATE = "plotly_dark"
COLOR_ACCENT = "#6366F1" # Indigo
COLOR_SUCCESS = "#10B981" # Green
COLOR_WARNING = "#F59E0B" # Orange
COLOR_DANGER = "#EF4444" # Red
COLOR_CYAN = "#06B6D4" # Cyan

def build_quality_scorecard_gauges(scorecard: Dict[str, Any]) -> gg.Figure:
    """Chart 1: Enterprise Data Quality Scorecard Gauges."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )

    metrics = [
        ("Overall Health", scorecard.get("overall_score", 0), scorecard.get("badge_color", COLOR_SUCCESS), 1),
        ("Completeness", scorecard.get("completeness_score", 0), COLOR_CYAN, 2),
        ("Accuracy", scorecard.get("accuracy_score", 0), COLOR_ACCENT, 3),
        ("Consistency", scorecard.get("consistency_score", 0), COLOR_WARNING, 4)
    ]

    for title, val, color, col in metrics:
        fig.add_trace(
            gg.Indicator(
                mode="gauge+number",
                value=val,
                title={'text': title, 'font': {'size': 14, 'color': '#F8FAFC'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'threshold': {'line': {'color': "#FFFFFF", 'width': 3}, 'thickness': 0.75, 'value': 80}
                }
            ),
            row=1, col=col
        )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def build_missing_values_heatmap(df_before: pd.DataFrame, df_after: pd.DataFrame = None) -> gg.Figure:
    """Chart 2: Missing Value Heatmap (Before/After comparison)."""
    if df_after is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Cleaning (Missing Nulls)", "After Cleaning"))
        
        # Before
        null_mask_b = df_before.isna().astype(int)
        fig.add_trace(gg.Heatmap(z=null_mask_b.values, x=df_before.columns, colorscale=[[0, "#1E1E2F"], [1, COLOR_DANGER]], showscale=False), row=1, col=1)
        
        # After
        null_mask_a = df_after.isna().astype(int)
        fig.add_trace(gg.Heatmap(z=null_mask_a.values, x=df_after.columns, colorscale=[[0, "#1E1E2F"], [1, COLOR_DANGER]], showscale=False), row=1, col=2)
        fig.update_layout(height=350)
    else:
        null_mask = df_before.isna().astype(int)
        fig = px.imshow(null_mask, labels=dict(x="Columns", y="Rows", color="Missing"), color_continuous_scale=[[0, "#1E1E2F"], [1, COLOR_DANGER]])
        fig.update_layout(title="Missing Values Heatmap (Red = Missing Cell)", height=320)

    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def build_duplicate_analysis_chart(scorecard: Dict[str, Any]) -> gg.Figure:
    """Chart 3: Duplicate Analysis Bar/Pie."""
    total_dupes = scorecard.get("total_duplicates", 0)
    unique_rows = max(1, scorecard.get("total_rows", 100) - total_dupes)

    fig = px.pie(
        names=["Unique Records", "Duplicate Records"],
        values=[unique_rows, total_dupes],
        color=["Unique Records", "Duplicate Records"],
        color_discrete_map={"Unique Records": COLOR_SUCCESS, "Duplicate Records": COLOR_DANGER},
        hole=0.4,
        title="Duplicate vs Unique Record Ratio"
    )
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=280)
    return fig

def build_outlier_boxplots(df: pd.DataFrame) -> gg.Figure:
    """Chart 4: Outlier Boxplots for Numeric Features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    if not num_cols:
        fig = gg.Figure()
        fig.add_annotation(text="No numeric columns for Outlier Boxplot", showarrow=False)
        return fig

    fig = gg.Figure()
    for col in num_cols:
        fig.add_trace(gg.Box(y=df[col].dropna(), name=col, marker_color=COLOR_ACCENT, boxmean='sd'))

    fig.update_layout(
        title="Numeric Outlier Boxplots (IQR & Standard Deviation)",
        template=DARK_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    return fig

def build_distribution_histogram(df: pd.DataFrame, selected_col: str) -> gg.Figure:
    """Chart 5: Data Distribution Histogram with KDE curve."""
    if selected_col not in df.columns or not pd.api.types.is_numeric_dtype(df[selected_col]):
        fig = gg.Figure()
        return fig

    fig = px.histogram(
        df, x=selected_col, marginal="box", color_discrete_sequence=[COLOR_CYAN],
        title=f"Value Distribution & Density: {selected_col}"
    )
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
    return fig

def build_data_types_donut(df: pd.DataFrame) -> gg.Figure:
    """Chart 6: Data Type Distribution Donut Chart."""
    dtypes_count = df.dtypes.astype(str).value_counts().reset_index()
    dtypes_count.columns = ["Data Type", "Count"]

    fig = px.pie(
        dtypes_count, names="Data Type", values="Count", hole=0.5,
        title="Data Type Distribution", color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=280)
    return fig

def build_cleaning_impact_chart(impact: Dict[str, Any]) -> gg.Figure:
    """Chart 7: Cleaning Impact Dual Bar Chart (Before vs After)."""
    categories = ["Missing Cells", "Duplicate Rows", "Outliers", "Quality Score"]
    before_vals = [impact.get("initial_missing", 50), impact.get("initial_dupes", 15), impact.get("initial_outliers", 10), impact.get("initial_score", 65)]
    after_vals = [impact.get("final_missing", 0), impact.get("final_dupes", 0), impact.get("final_outliers", 0), impact.get("final_score", 95)]

    fig = gg.Figure(data=[
        gg.Bar(name='Before Cleaning', x=categories, y=before_vals, marker_color=COLOR_DANGER),
        gg.Bar(name='After Cleaning', x=categories, y=after_vals, marker_color=COLOR_SUCCESS)
    ])
    fig.update_layout(
        barmode='group',
        title="Data Cleaning Impact Comparison (Before vs After)",
        template=DARK_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320
    )
    return fig

def build_correlation_heatmap(df: pd.DataFrame) -> gg.Figure:
    """Chart 8: Pairwise Correlation Heatmap."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        fig = gg.Figure()
        fig.add_annotation(text="At least 2 numeric columns required for correlation heatmap", showarrow=False)
        return fig

    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Pairwise Feature Correlation Heatmap")
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=350)
    return fig

def build_pattern_frequency_chart(pattern_matches: Dict[str, Any]) -> gg.Figure:
    """Chart 9: Regex Text Pattern Frequency Analysis."""
    p_data = []
    for col, matches in pattern_matches.items():
        for pat_name, cnt in matches.items():
            p_data.append({"Column": col, "Pattern": pat_name, "Matches": cnt})

    if not p_data:
        fig = gg.Figure()
        fig.add_annotation(text="No special regex patterns detected", showarrow=False)
        return fig

    df_p = pd.DataFrame(p_data)
    fig = px.bar(df_p, x="Column", y="Matches", color="Pattern", title="Detected Text Field Regex Patterns", barmode="group")
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=320)
    return fig

def build_quality_trend_chart(audit_trail: pd.DataFrame) -> gg.Figure:
    """Chart 10: Data Quality Score Trend Over Pipeline Steps."""
    if audit_trail is None or audit_trail.empty:
        fig = gg.Figure()
        return fig

    fig = px.line(audit_trail, x="Step #", y="Rows", markers=True, title="Row Volume Across Cleaning Operations")
    fig.update_traces(line_color=COLOR_ACCENT, line_width=3)
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=280)
    return fig

def build_anomaly_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str) -> gg.Figure:
    """Chart 11: Anomaly Detection Scatter Plot."""
    if x_col not in df.columns or y_col not in df.columns:
        fig = gg.Figure()
        return fig

    fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[COLOR_CYAN], title=f"Anomaly Distribution: {x_col} vs {y_col}")
    fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=320)
    return fig

def build_lineage_flow_chart(pipeline_steps: List[Dict[str, Any]]) -> gg.Figure:
    """Chart 12: Pipeline Lineage Flow Chart."""
    step_names = [s.get("action", "Step") for s in pipeline_steps if s.get("enabled", True)]
    if not step_names:
        step_names = ["Load Data", "Clean Nulls", "Deduplicate", "Report"]

    fig = gg.Figure(data=[gg.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=step_names, color=[COLOR_ACCENT] * len(step_names)
        ),
        link=dict(
            source=list(range(len(step_names) - 1)),
            target=list(range(1, len(step_names))),
            value=[100] * (len(step_names) - 1)
        )
    )])
    fig.update_layout(title="ETL Pipeline Lineage Flow", template=DARK_TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=260)
    return fig
