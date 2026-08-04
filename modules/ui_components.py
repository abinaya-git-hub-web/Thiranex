"""
Module: ui_components.py
Description: Custom HTML UI components, metric card generators, header banners, 
             onboarding tour guide modal, FAQ sidebar, theme injector, and footer.
"""

import os
import streamlit as st
from typing import Dict, Any


def inject_custom_css(css_file_path: str = "assets/style.css"):
    """
    Injects external CSS styling into Streamlit application DOM.

    Args:
        css_file_path (str): Relative or absolute path to style.css.
    """
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header_banner(active_filter_meta: Dict[str, Any]):
    """
    Renders enterprise header banner with branding logo, title, and active filter pill.

    Args:
        active_filter_meta (Dict[str, Any]): Active filter count and summary badges.
    """
    filter_count = active_filter_meta.get("count", 0)
    filter_pill_html = ""
    if filter_count > 0:
        filter_pill_html = f"""
        <div class="tx-filter-pill">
            <span>⚡ {filter_count} Active Filter{'s' if filter_count > 1 else ''}</span>
        </div>
        """

    logo_path = "assets/logo.svg"
    logo_svg = ""
    if os.path.exists(logo_path):
        with open(logo_path, "r", encoding="utf-8") as f:
            logo_svg = f.read()

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f"""
            <div class="tx-header-banner">
                <div style="display: flex; align-items: center; gap: 1.25rem;">
                    <div style="width: 180px; height: 50px;">
                        {logo_svg}
                    </div>
                    <div>
                        <h1 class="tx-header-title">Thiranex Solutions</h1>
                        <p class="tx-header-subtitle">Enterprise Sales & Revenue Intelligence Platform</p>
                    </div>
                </div>
                {filter_pill_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Theme Mode Selector / Reset Quick Button
        st.write("")


def render_kpi_card(title: str, value: str, delta_pct: float, subtext: str = "vs prior period", icon: str = "📊"):
    """
    Renders styled Glassmorphism HTML KPI card with trend indicator badge.

    Args:
        title (str): Metric title.
        value (str): Main metric value display string.
        delta_pct (float): Percentage change versus prior period.
        subtext (str): Subtitle description.
        icon (str): Emoji icon representation.
    """
    if delta_pct > 0:
        badge_class = "tx-badge-positive"
        arrow = "▲"
        delta_str = f"+{delta_pct:.1f}%"
    elif delta_pct < 0:
        badge_class = "tx-badge-negative"
        arrow = "▼"
        delta_str = f"{delta_pct:.1f}%"
    else:
        badge_class = "tx-badge-neutral"
        arrow = "►"
        delta_str = "0.0%"

    card_html = f"""
    <div class="tx-glass-card">
        <div class="tx-kpi-header">
            <span class="tx-kpi-title">{icon} {title}</span>
            <span class="tx-kpi-badge {badge_class}">{arrow} {delta_str}</span>
        </div>
        <div class="tx-kpi-value">{value}</div>
        <div class="tx-kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_onboarding_tour():
    """
    Displays an interactive quick onboarding tour for first-time dashboard visitors.
    """
    with st.expander("👋 New to Thiranex Analytics? Take a 1-Minute Dashboard Tour", expanded=False):
        st.markdown("""
        ### Welcome to Thiranex Solutions Sales Intelligence!
        Here is how to get the most out of your dashboard:

        1. **📁 Data Import & Detection**: Upload your own CSV, Excel, or JSON sales file in the sidebar. Our smart engine auto-detects column types.
        2. **📊 Interactive Charts**: Hover over chart data points, double-click legends to isolate categories, and toggle full-screen views.
        3. **🤖 AI Insights**: Navigate to the **AI Insights** tab to view Isolation Forest anomaly detection and 30-day forecast predictions.
        4. **📑 PDF & CSV Exports**: Use the **Export & Reports** tab to generate official PDF executive summaries or download filtered CSV datasets.
        5. **🎛️ Dynamic Filters**: Use the left sidebar to slice data by Date Range, Product Category, Region, Salesperson, and Price Sliders.
        """)


def render_help_sidebar_faq():
    """
    Renders Help Sidebar FAQs expander section.
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("❓ Help & FAQs"):
        st.markdown("""
        **Q: How does Column Auto-Detection work?**  
        *A: Our system scans column headers and data types to map Revenue, Dates, Quantities, and Products automatically.*

        **Q: What is Isolation Forest Anomaly Detection?**  
        *A: An unsupervised ML model that identifies statistical outliers (unusual spikes or sharp drops in transactions).*

        **Q: How is the 30-day forecast computed?**  
        *A: Uses Holt-Winters Exponential Smoothing on historical daily trend data.*
        """)


def render_footer():
    """
    Renders persistent footer signature.
    """
    footer_html = """
    <div class="tx-footer">
        <p>© 2026 <b>Thiranex Solutions</b> — Built with ❤️ using Streamlit, Plotly & Scikit-Learn</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
