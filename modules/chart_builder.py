"""
Module: chart_builder.py
Description: Interactive Plotly Data Visualization Engine for Customer Segmentation.
             Implements 12+ glassmorphic dark-themed charts with fail-safe HTML fallback.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List
from scipy.cluster.hierarchy import dendrogram


# Unified Thiranex Dark Glassmorphism Color Palette
THEME_COLORS = [
    "#8B5CF6", "#3B82F6", "#10B981", "#F59E0B", "#EC4899",
    "#06B6D4", "#6366F1", "#A855F7", "#14B8A6", "#EAB308"
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.7)",
    plot_bgcolor="rgba(15, 23, 42, 0.7)",
    font=dict(family="Inter, Roboto, sans-serif", color="#F8FAFC", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(30, 41, 59, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.1)",
        borderwidth=1,
        font=dict(color="#CBD5E1")
    )
)


def _apply_theme(fig: go.Figure, title: str):
    """Applies common dark theme styling and responsive layout to a Plotly figure."""
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8")
        )
    )


def render_plotly_chart(fig: go.Figure, height: int = 450, key: str = None):
    """
    Renders Plotly chart with Streamlit st.plotly_chart and automatic HTML CDN fallback.
    Prevents JS chunk loading errors in modern Streamlit environments.
    """
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception:
        html_str = fig.to_html(include_plotlyjs="cdn", full_html=False)
        st.components.v1.html(html_str, height=height, scrolling=False)


# 1. Segment Distribution Donut Chart
def build_segment_distribution_chart(df: pd.DataFrame, segment_col: str) -> go.Figure:
    """Builds a sleek donut chart showing the percentage breakdown of customer segments."""
    counts = df[segment_col].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Segment",
        hole=0.55,
        color_discrete_sequence=THEME_COLORS
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#0F172A", width=2))
    )
    _apply_theme(fig, "📊 Segment Distribution Breakdown")
    return fig


# 2. Segment Profile Comparison Radar Chart
def build_radar_comparison_chart(df: pd.DataFrame, segment_col: str, mappings: Dict[str, str]) -> go.Figure:
    """Builds a Radar Chart comparing average normalized metrics across segments."""
    metrics = ["Age", "Income", "Total Spend ($)", "Purchase Frequency", "Average Order Value ($)", "Recency_Days"]
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 3:
        available_metrics = [c for c in ["Monetary_Val", "Frequency_Val", "Recency_Days", "Age", "Income"] if c in df.columns]

    grouped = df.groupby(segment_col)[available_metrics].mean()
    normalized = (grouped - grouped.min()) / (grouped.max() - grouped.min() + 1e-6)

    fig = go.Figure()
    categories = available_metrics

    for idx, (segment, row) in enumerate(normalized.iterrows()):
        values = row.values.tolist()
        values.append(values[0])
        cat_loop = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cat_loop,
            fill="toself",
            name=str(segment),
            line=dict(color=THEME_COLORS[idx % len(THEME_COLORS)], width=2),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255, 255, 255, 0.1)", showticklabels=False),
            angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#CBD5E1"))
        )
    )
    _apply_theme(fig, "🕸️ Multi-Dimensional Segment Comparison (Radar)")
    return fig


# 3. Demographic Heatmap (Age vs Income colored by Segment)
def build_demographic_heatmap(df: pd.DataFrame, segment_col: str, mappings: Dict[str, str]) -> go.Figure:
    """Scatter heatmap showing Age vs Income distribution with segment color coding."""
    age_c = mappings.get("age") if mappings.get("age") in df.columns else "Age"
    inc_c = mappings.get("income") if mappings.get("income") in df.columns else "Income"

    if age_c not in df.columns or inc_c not in df.columns:
        fig = go.Figure()
        _apply_theme(fig, "Demographic Heatmap (Age/Income columns missing)")
        return fig

    fig = px.scatter(
        df,
        x=age_c,
        y=inc_c,
        color=segment_col,
        size="Total Spend ($)" if "Total Spend ($)" in df.columns else None,
        hover_data=["Customer ID"] if "Customer ID" in df.columns else None,
        color_discrete_sequence=THEME_COLORS,
        opacity=0.75
    )
    _apply_theme(fig, "🔥 Demographic Matrix: Age vs Income by Segment")
    fig.update_xaxes(title="Customer Age (Years)")
    fig.update_yaxes(title="Annual Income ($)")
    return fig


# 4. Spending Pattern Box Plot per Segment
def build_spending_boxplot(df: pd.DataFrame, segment_col: str, mappings: Dict[str, str]) -> go.Figure:
    """Box plots showing spend distribution per segment."""
    spend_c = mappings.get("spend") if mappings.get("spend") in df.columns else ("Total Spend ($)" if "Total Spend ($)" in df.columns else "Monetary_Val")

    fig = px.box(
        df,
        x=segment_col,
        y=spend_c,
        color=segment_col,
        points="outliers",
        color_discrete_sequence=THEME_COLORS
    )
    _apply_theme(fig, "💰 Total Spend Distribution per Segment")
    fig.update_xaxes(title="Segment")
    fig.update_yaxes(title="Total Spend ($)")
    return fig


# 5. Geographic Segment Distribution Bar Chart
def build_geographic_chart(df: pd.DataFrame, segment_col: str, mappings: Dict[str, str]) -> go.Figure:
    """Grouped bar chart showing segment proportions across locations/regions."""
    loc_c = mappings.get("location") if mappings.get("location") in df.columns else "Location"

    if loc_c not in df.columns:
        fig = go.Figure()
        _apply_theme(fig, "Geographic Map (Location column missing)")
        return fig

    geo_df = df.groupby([loc_c, segment_col]).size().reset_index(name="Customer_Count")

    fig = px.bar(
        geo_df,
        x=loc_c,
        y="Customer_Count",
        color=segment_col,
        barmode="group",
        color_discrete_sequence=THEME_COLORS
    )
    _apply_theme(fig, "🗺️ Geographic Regional Segment Distribution")
    fig.update_xaxes(title="Region / Location")
    fig.update_yaxes(title="Customer Count")
    return fig


# 6. RFM 3D Scatter Plot
def build_rfm_3d_scatter(df: pd.DataFrame, rfm_col: str = "RFM_Segment") -> go.Figure:
    """Interactive 3D Scatter Plot mapping Recency vs Frequency vs Monetary scores."""
    r_col = "Recency_Days" if "Recency_Days" in df.columns else "R_Score"
    f_col = "Frequency_Val" if "Frequency_Val" in df.columns else "F_Score"
    m_col = "Monetary_Val" if "Monetary_Val" in df.columns else "M_Score"

    seg_c = rfm_col if rfm_col in df.columns else "KMeans_Cluster" if "KMeans_Cluster" in df.columns else None

    fig = px.scatter_3d(
        df,
        x=r_col,
        y=f_col,
        z=m_col,
        color=seg_c,
        hover_name="Customer ID" if "Customer ID" in df.columns else None,
        color_discrete_sequence=THEME_COLORS,
        opacity=0.8
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Recency (Days)", backgroundcolor="#0F172A", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Frequency (Orders)", backgroundcolor="#0F172A", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Monetary Spend ($)", backgroundcolor="#0F172A", gridcolor="rgba(255,255,255,0.1)")
        )
    )
    _apply_theme(fig, "🧊 3D RFM Matrix (Recency vs Frequency vs Monetary)")
    return fig


# 7. Customer Lifecycle Sankey Diagram
def build_lifecycle_sankey(df: pd.DataFrame, segment_col: str) -> go.Figure:
    """Sankey diagram showing flow from Location -> Segment -> Preferred Payment."""
    loc_c = "Location" if "Location" in df.columns else None
    pay_c = "Preferred Payment" if "Preferred Payment" in df.columns else None

    if not loc_c or not pay_c or segment_col not in df.columns:
        fig = go.Figure()
        _apply_theme(fig, "Customer Lifecycle Flow")
        return fig

    locations = list(df[loc_c].unique())
    segments = list(df[segment_col].unique())
    payments = list(df[pay_c].unique())

    all_nodes = locations + [str(s) for s in segments] + payments
    node_map = {n: i for i, n in enumerate(all_nodes)}

    flow1 = df.groupby([loc_c, segment_col]).size().reset_index(name="count")
    sources = [node_map[r[loc_c]] for _, r in flow1.iterrows()]
    targets = [node_map[str(r[segment_col])] for _, r in flow1.iterrows()]
    values = flow1["count"].tolist()

    flow2 = df.groupby([segment_col, pay_c]).size().reset_index(name="count")
    sources += [node_map[str(r[segment_col])] for _, r in flow2.iterrows()]
    targets += [node_map[r[pay_c]] for _, r in flow2.iterrows()]
    values += flow2["count"].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#0F172A", width=1),
            label=all_nodes,
            color=["#8B5CF6" if n in locations else "#3B82F6" if n in payments else "#10B981" for n in all_nodes]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(139, 92, 246, 0.2)"
        )
    )])
    _apply_theme(fig, "🔀 Customer Lifecycle Journey (Location ➔ Segment ➔ Payment)")
    return fig


# 8. Hierarchical Dendrogram Chart
def build_dendrogram_chart(linkage_matrix: np.ndarray) -> go.Figure:
    """Converts Scipy linkage matrix into a Plotly interactive dendrogram figure."""
    dendro = dendrogram(linkage_matrix, no_plot=True)

    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])

    fig = go.Figure()
    for i, d in zip(icoord, dcoord):
        fig.add_trace(go.Scatter(
            x=i, y=d,
            mode='lines',
            line=dict(color="#8B5CF6", width=1.5),
            hoverinfo='none',
            showlegend=False
        ))

    _apply_theme(fig, "🌳 Hierarchical Clustering Dendrogram")
    fig.update_xaxes(title="Customer Index Clusters", showticklabels=False)
    fig.update_yaxes(title="Distance Threshold (Euclidean)")
    return fig


# 9. Elbow Curve Chart
def build_elbow_chart(k_values: List[int], inertias: List[float]) -> go.Figure:
    """Builds line plot for determining optimal cluster count (K) via Elbow method."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertias,
        mode="lines+markers",
        name="WCSS (Inertia)",
        line=dict(color="#8B5CF6", width=3),
        marker=dict(size=8, color="#3B82F6")
    ))
    _apply_theme(fig, "📐 K-Means Elbow Method (Inertia vs K)")
    fig.update_xaxes(title="Number of Clusters (K)")
    fig.update_yaxes(title="Within-Cluster Sum of Squares (WCSS)")
    return fig


# 10. Silhouette Score Chart
def build_silhouette_chart(k_values: List[int], silhouette_scores: List[float]) -> go.Figure:
    """Builds bar chart evaluating Silhouette score across different cluster counts."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_values,
        y=silhouette_scores,
        name="Silhouette Score",
        marker=dict(color=THEME_COLORS[:len(k_values)])
    ))
    _apply_theme(fig, "📊 Silhouette Score Evaluation per K")
    fig.update_xaxes(title="Number of Clusters (K)")
    fig.update_yaxes(title="Silhouette Coefficient")
    return fig


# 11. Churn Drivers Feature Importance Chart
def build_churn_driver_chart(drivers: List[tuple]) -> go.Figure:
    """Bar chart showing top features driving customer churn."""
    features = [d[0] for d in drivers]
    importances = [d[1] for d in drivers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation="h",
        marker=dict(color="#EC4899")
    ))
    _apply_theme(fig, "🚨 Churn Drivers (Random Forest Feature Importances)")
    fig.update_xaxes(title="Importance Weight")
    fig.update_yaxes(title="Feature", autorange="reversed")
    return fig


# 12. CLV Tier Distribution Chart
def build_clv_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart displaying CLV Tier breakdown."""
    if "CLV_Tier" not in df.columns:
        fig = go.Figure()
        _apply_theme(fig, "CLV Tiers")
        return fig

    counts = df["CLV_Tier"].value_counts().reset_index()
    counts.columns = ["Tier", "Count"]

    fig = px.bar(
        counts,
        x="Tier",
        y="Count",
        color="Tier",
        color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B"]
    )
    _apply_theme(fig, "💎 Customer Lifetime Value (CLV) Tiers Distribution")
    fig.update_xaxes(title="CLV Tier")
    fig.update_yaxes(title="Customer Count")
    return fig
