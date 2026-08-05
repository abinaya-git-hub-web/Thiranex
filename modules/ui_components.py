"""
Module: ui_components.py
Description: Custom CSS Theme Engine (Dark Glassmorphism), KPI Cards, Persona Viewers, 
             Side-by-Side Segment Comparers, Onboarding Tour, and Footer.
"""

import streamlit as st
from typing import Dict, Any, List


def inject_custom_css():
    """Injects custom CSS styling for dark theme glassmorphism and modern UI aesthetics."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #8B5CF6 !important;
    }

    /* Header Banner Styling */
    .tx-header-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .tx-header-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A855F7, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .tx-header-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Metric / KPI Card Styling */
    .tx-kpi-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .tx-kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139, 92, 246, 0.5);
    }
    .tx-kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .tx-kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0.3rem 0;
    }

    /* Persona Card Glassmorphism */
    .tx-persona-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
    }
    .tx-persona-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.8rem;
        margin-bottom: 1rem;
    }
    .tx-persona-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .tx-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        border-radius: 10px;
        background-color: rgba(30, 41, 59, 0.5);
        color: #94A3B8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        border-color: #8B5CF6 !important;
        box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.39);
    }

    /* Footer */
    .tx-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        color: #64748B;
        font-size: 0.85rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header_banner(meta: Dict[str, Any]):
    """Renders the top executive header banner."""
    seg_method = meta.get("segmentation_method", "K-Means Clustering")
    filtered_cnt = meta.get("filtered_records", 1000)
    total_cnt = meta.get("total_records", 1000)

    html = f"""
    <div class="tx-header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="tx-header-title">Thiranex Solutions — Customer Segmentation Engine</h1>
                <div class="tx-header-subtitle">Enterprise Customer Persona & Machine Learning Intelligence Dashboard</div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span style="background: rgba(139, 92, 246, 0.2); color: #A855F7; border: 1px solid rgba(139, 92, 246, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600; margin-right: 8px;">
                    🧠 {seg_method}
                </span>
                <span style="background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                    👥 {filtered_cnt:,} / {total_cnt:,} Active Records
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, delta: str = "", caption: str = "", icon: str = "📊"):
    """Renders a single KPI metric card."""
    html = f"""
    <div class="tx-kpi-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{icon}</div>
        <div class="tx-kpi-title">{title}</div>
        <div class="tx-kpi-value">{value}</div>
        <div style="color: #94A3B8; font-size: 0.75rem;">{caption}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_persona_card(persona: Dict[str, Any]):
    """Renders a detailed customer persona card."""
    c = persona["theme_color"]

    html = f"""
    <div class="tx-persona-card" style="border-left: 5px solid {c};">
        <div class="tx-persona-header">
            <div>
                <h3 class="tx-persona-name" style="color: {c};">{persona['creative_name']}</h3>
                <span style="color: #94A3B8; font-size: 0.85rem;">Raw Segment: {persona['segment_id']} | {persona['count']:,} Customers ({persona['percentage']}%)</span>
            </div>
            <div>
                <span class="tx-badge" style="background: rgba(255,255,255,0.08); color: #F8FAFC; border: 1px solid {c};">{persona['risk_badge']}</span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
            <div style="background: rgba(15, 23, 42, 0.5); padding: 0.8rem; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">👤 DEMOGRAPHICS</div>
                <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;"><b>Age:</b> {persona['demographics']['age_range']}</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Avg Income:</b> {persona['demographics']['avg_income']}</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Top Location:</b> {persona['demographics']['top_location']}</div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.5); padding: 0.8rem; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">💰 MONETARY PROFILE</div>
                <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;"><b>Avg Spend:</b> ${persona['avg_spend']:,.2f}</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Avg Order Value:</b> ${persona['avg_aov']:,.2f}</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Revenue Share:</b> {persona['revenue_share']}%</div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.5); padding: 0.8rem; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">🛒 BEHAVIORAL TRAITS</div>
                <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;"><b>Purchase Freq:</b> {persona['avg_frequency']} times/yr</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Categories:</b> {persona['behavioral']['top_categories']}</div>
                <div style="font-size: 0.85rem; color: #E2E8F0;"><b>Payment:</b> {persona['behavioral']['preferred_payment']}</div>
            </div>
        </div>

        <div style="background: rgba(139, 92, 246, 0.1); border: 1px dashed rgba(139, 92, 246, 0.3); padding: 0.8rem; border-radius: 10px;">
            <div style="font-size: 0.85rem; color: #A855F7;"><b>🎯 Marketing Strategy:</b> {persona['strategy']}</div>
            <div style="font-size: 0.85rem; color: #10B981; margin-top: 4px;"><b>📣 Recommended Offer:</b> {persona['recommended_offers']}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_onboarding_tour():
    """Renders expandable user guide modal for dashboard onboarding."""
    with st.expander("❓ Quick Start Guide & App Walkthrough", expanded=False):
        st.markdown(
            """
            #### 🚀 Welcome to the Thiranex Solutions Customer Segmentation Suite!
            
            This application uses Machine Learning & RFM analysis to group customers by demographic & behavioral patterns:
            
            1. **Select Data Source**: Use synthetic demo data (1,000 customers with 3 natural segments) or upload a custom CSV/Excel/JSON file via the sidebar.
            2. **Choose Segmentation Algorithm**:
               - **K-Means Clustering (ML)**: Groups customers using normalized spend, frequency, AOV, recency, age, and income.
               - **RFM Analysis (Traditional)**: Evaluates Recency, Frequency, and Monetary scores (1-5 scale).
               - **DBSCAN Clustering**: Identifies irregular clusters and density outliers.
               - **Hierarchical Clustering**: Agglomerative grouping with Dendrogram tree.
            3. **Explore Customer Personas**: View auto-generated demographic, monetary, and behavioral profiles for each segment.
            4. **Predict Churn & CLV**: Use Random Forest models to predict churn risk drivers and projected Customer Lifetime Value.
            5. **Export PDF Reports & Segmented CSV**: Download executive PDF persona decks and raw data with segment tags.
            """
        )


def render_footer():
    """Renders the bottom copyright footer."""
    st.markdown(
        """
        <div class="tx-footer">
            © 2026 <b>Thiranex Solutions</b> — Enterprise Machine Learning Intelligence Studio | Built with ❤️ using Streamlit, Scikit-Learn & Plotly
        </div>
        """,
        unsafe_allow_html=True
    )
