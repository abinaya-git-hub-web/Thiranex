"""
=============================================================================
Thiranex Solutions — Modular UI Component Library
Royal Purple Theme Edition
=============================================================================
"""

import streamlit as st
import os

def inject_custom_css():
    """Injects custom Royal Purple stylesheet."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background: #F8F7FC; color: #1F1B2D; }
        </style>
        """, unsafe_allow_html=True)

def render_header_banner():
    """Renders top hero header banner with Thiranex Solutions Royal Purple branding."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(109, 40, 217, 0.08) 0%, rgba(124, 58, 237, 0.04) 100%);
        border: 1px solid #E5E0F0;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(109, 40, 217, 0.05);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #6D28D9; letter-spacing: 0.1em; text-transform: uppercase;">
                    ⚡ THIRANEX SOLUTIONS ENTERPRISE SUITE
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #1F1B2D; margin-top: 0.2rem; letter-spacing: -0.02em;">
                    Data Cleaning & Reporting Automation Platform
                </div>
                <div style="font-size: 0.95rem; color: #6B6478; margin-top: 0.3rem;">
                    Autonomous ETL Engine • Quality Profiling • AI Imputation • Multi-Format Reporting (PDF, Excel, HTML)
                </div>
            </div>
            <div style="text-align: right; margin-top: 0.5rem;">
                <span class="status-badge-primary">LIVE PRODUCTION ENGINE</span>
                <div style="font-size: 0.75rem; color: #6B6478; margin-top: 0.4rem; font-weight: 500;">v2.5.0 Enterprise</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_step_progress(current_step: int):
    """Renders visual 8-step workflow progress bar with Royal Purple theme."""
    steps = [
        "1. Ingest", "2. Profile", "3. Strategy", "4. Clean",
        "5. Verify", "6. Format", "7. Export", "8. Schedule"
    ]
    cols = st.columns(8)
    for idx, (col, step_label) in enumerate(zip(cols, steps), start=1):
        with col:
            if idx < current_step:
                st.markdown(f"<div style='text-align:center; padding:8px; background:rgba(22,163,74,0.1); border:1px solid #16A34A; border-radius:8px; font-weight:bold; color:#16A34A; font-size:0.75rem;'>✓ {step_label}</div>", unsafe_allow_html=True)
            elif idx == current_step:
                st.markdown(f"<div style='text-align:center; padding:8px; background:linear-gradient(135deg, #6D28D9, #7C3AED); border-radius:8px; font-weight:bold; color:#FFFFFF; font-size:0.75rem; box-shadow:0 4px 12px rgba(109,40,217,0.3);'>▶ {step_label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center; padding:8px; background:#FFFFFF; border:1px solid #E5E0F0; border-radius:8px; color:#6B6478; font-size:0.75rem;'>{step_label}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, subtext: str = "", badge: str = "Good"):
    """Renders a Royal Purple KPI card."""
    badge_cls = f"status-badge-{badge.lower()}"
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="margin-top: 0.4rem; font-size: 0.8rem; color: #6B6478;">{subtext}</div>
        <div style="margin-top: 0.6rem;"><span class="{badge_cls}">{badge.upper()}</span></div>
    </div>
    """, unsafe_allow_html=True)

def render_onboarding_tour():
    """Renders interactive onboarding guide expander."""
    with st.expander("🚀 Platform Onboarding Guide & Guided 8-Step User Workflow", expanded=False):
        st.markdown("""
        ### Thiranex Enterprise 8-Step Workflow Walkthrough:
        
        1. **Step 1: Data Integration**: Import datasets via upload, database, cloud bucket, REST API, or generate synthetic demo data.
        2. **Step 2: Auto-Profiling & Scorecard**: Review data types, null counts, regex pattern breakdowns, and the **Data Health Score (0-100)**.
        3. **Step 3: Cleaning Strategy Selection**: Select **Auto-Clean (recommended)**, **Manual Customization**, or **Visual Pipeline Builder**.
        4. **Step 4: Apply Cleaning & Compare**: Execute transformation and inspect real-time before/after impact charts.
        5. **Step 5: Result Verification**: Verify clean rows, fine-tune filters, and inspect the step-by-step audit trail.
        6. **Step 6: Report Format Selection**: Select Executive Dashboard, Technical Log, or Comparative Briefing in PDF, Excel, HTML, or JSON.
        7. **Step 7: Generate & Export**: One-click download of branded ReportLab PDFs, multi-tab OpenPyXL Excel workbooks, or clean data.
        8. **Step 8: Automated Scheduling**: Schedule recurring Daily/Weekly/Monthly jobs, folder watchers, and stakeholder email alerts.
        """)

def render_footer():
    """Renders platform footer."""
    st.markdown("""
    <hr style="border: none; border-top: 1px solid #E5E0F0; margin-top: 3rem; margin-bottom: 1.5rem;">
    <div style="text-align: center; color: #6B6478; font-size: 0.85rem;">
        © 2026 <b>Thiranex Solutions</b> — Built with ❤️ for Enterprise ETL, Quality Profiling & Automation.
    </div>
    """, unsafe_allow_html=True)
