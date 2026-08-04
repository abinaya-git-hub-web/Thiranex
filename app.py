"""
=============================================================================
Thiranex Solutions — Enterprise Sales & Revenue Analysis Dashboard
Author: Google Deepmind Agentic AI Team
Technology Stack: Streamlit, Plotly, Pandas, Scikit-Learn, Statsmodels, ReportLab
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import Modular Engine Components
from modules.data_loader import generate_sample_data, load_uploaded_file
from modules.preprocessor import detect_column_types, standardize_and_clean_data, generate_data_profile
from modules.kpi_calculator import calculate_kpis
from modules import chart_builder as cb
from modules import ai_insights as ai
from modules.filters import render_sidebar_filters
from modules import exporters
from modules import ui_components as ui

# Page Configuration
st.set_page_config(
    page_title="Thiranex Solutions — Sales & Revenue Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS Styling
ui.inject_custom_css()


def main():
    """Main Application Execution Function."""
    
    # ---------------------------------------------------------
    # 1. Sidebar Data Import & Detection
    # ---------------------------------------------------------
    st.sidebar.image("assets/logo.svg", width=200) if False else None
    st.sidebar.markdown("### 📁 Data Source Selection")

    data_option = st.sidebar.radio(
        "Select Data Source",
        ["Use Synthetic Demo Data (500 rows)", "Upload Custom Dataset (CSV, Excel, JSON)"],
        key="sb_data_option"
    )

    df_raw = None
    upload_error = None

    if data_option == "Upload Custom Dataset (CSV, Excel, JSON)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Sales File",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )
        if uploaded_file is not None:
            with st.spinner("⏳ Parsing uploaded dataset..."):
                df_raw, upload_error = load_uploaded_file(uploaded_file)
            if upload_error:
                st.error(f"❌ {upload_error}")
                st.stop()
        else:
            st.info("👆 Please upload a sales data file to proceed, or switch to Demo Data.")
            st.stop()
    else:
        with st.spinner("⚡ Generating Thiranex synthetic enterprise sales data..."):
            df_raw = generate_sample_data(num_rows=500)

    # ---------------------------------------------------------
    # 2. Smart Column Detection & Preprocessing
    # ---------------------------------------------------------
    detected_mappings = detect_column_types(df_raw)
    cleaned_df = standardize_and_clean_data(df_raw, detected_mappings)

    # ---------------------------------------------------------
    # 3. Dynamic Sidebar Filters
    # ---------------------------------------------------------
    filtered_df, filter_meta = render_sidebar_filters(cleaned_df, detected_mappings)

    # Calculate KPIs
    kpis = calculate_kpis(filtered_df, detected_mappings)

    # ---------------------------------------------------------
    # 4. Header Banner & Onboarding Guide
    # ---------------------------------------------------------
    ui.render_header_banner(filter_meta)
    ui.render_onboarding_tour()

    # ---------------------------------------------------------
    # 5. Main Dashboard Tabs Layout
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Dashboard", 
        "🤖 AI Insights & Forecasting", 
        "🔍 Data Profiling & Explorer", 
        "📥 Export & Reports"
    ])

    # =========================================================
    # TAB 1: EXECUTIVE DASHBOARD
    # =========================================================
    with tab1:
        # Row 1: KPI Cards Grid
        st.markdown("#### 🚀 Key Performance Indicators")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            ui.render_kpi_card(
                "Total Revenue",
                f"${kpis['total_revenue']:,.2f}",
                kpis['revenue_delta'],
                "vs prior period",
                "💰"
            )
        with col2:
            ui.render_kpi_card(
                "Total Orders",
                f"{kpis['total_orders']:,}",
                kpis['orders_delta'],
                "vs prior period",
                "📦"
            )
        with col3:
            ui.render_kpi_card(
                "Avg Order Value",
                f"${kpis['aov']:,.2f}",
                kpis['aov_delta'],
                "AOV",
                "💳"
            )
        with col4:
            ui.render_kpi_card(
                "Units Sold",
                f"{kpis['units_sold']:,}",
                kpis['units_delta'],
                "volume",
                "📈"
            )
        with col5:
            ui.render_kpi_card(
                "Active Clients",
                f"{kpis['active_customers']:,}",
                kpis['customers_delta'],
                "distinct clients",
                "👥"
            )
        with col6:
            top_rev = kpis.get('top_product_revenue', 0.0)
            ui.render_kpi_card(
                "Top Seller",
                kpis['top_product'],
                0.0,
                f"${top_rev:,.0f} rev",
                "🏆"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Revenue Trend & Category Donut Chart
        r2_col1, r2_col2 = st.columns([2, 1])
        with r2_col1:
            fig_trend = cb.build_revenue_trend_chart(filtered_df, detected_mappings)
            st.plotly_chart(fig_trend, use_container_width=True)
        with r2_col2:
            fig_donut = cb.build_category_donut_chart(filtered_df, detected_mappings)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Row 3: Product Performance Matrix & Leaderboard
        r3_col1, r3_col2 = st.columns([1, 1])
        with r3_col1:
            fig_matrix = cb.build_product_performance_matrix(filtered_df, detected_mappings)
            st.plotly_chart(fig_matrix, use_container_width=True)
        with r3_col2:
            fig_lead = cb.build_salesperson_leaderboard(filtered_df, detected_mappings)
            st.plotly_chart(fig_lead, use_container_width=True)

        # Row 4: Calendar Heatmap & Geographic Map
        r4_col1, r4_col2 = st.columns([1, 1])
        with r4_col1:
            fig_heatmap = cb.build_monthly_heatmap(filtered_df, detected_mappings)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with r4_col2:
            fig_geo = cb.build_geographic_map(filtered_df, detected_mappings)
            st.plotly_chart(fig_geo, use_container_width=True)

    # =========================================================
    # TAB 2: AI INSIGHTS & FORECASTING
    # =========================================================
    with tab2:
        st.markdown("### 🤖 AI-Powered Intelligence Suite")

        # 1. Isolation Forest Anomaly Detection
        anomalies_df = ai.detect_sales_anomalies(filtered_df, detected_mappings)
        
        # 2. 30-Day Predictive Sales Forecast
        forecast_df = ai.generate_sales_forecast(filtered_df, detected_mappings, forecast_days=30)

        # 3. Natural Language Executive Summary Box
        nlg_summary = ai.generate_executive_summary(kpis, len(anomalies_df), forecast_df)
        st.markdown(
            f"""
            <div class="tx-ai-summary-box">
                <h4>✨ AI Automated Executive Narrative</h4>
                <div>{nlg_summary}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Forecasting Visuals
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            st.markdown("#### 🔮 30-Day Sales Revenue Forecast")
            if not forecast_df.empty:
                import plotly.graph_objects as go
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=forecast_df["Date"], y=forecast_df["Predicted_Revenue"],
                    mode="lines+markers", name="Predicted Revenue ($)",
                    line=dict(color="#8B5CF6", width=3)
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast_df["Date"], y=forecast_df["Upper_Bound"],
                    mode="lines", name="Upper 95% Confidence",
                    line=dict(color="rgba(139,92,246,0.2)", dash="dash")
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast_df["Date"], y=forecast_df["Lower_Bound"],
                    mode="lines", name="Lower 95% Confidence",
                    fill="tonexty", fillcolor="rgba(139,92,246,0.1)",
                    line=dict(color="rgba(139,92,246,0.2)", dash="dash")
                ))
                cb._apply_theme(fig_f, "Predicted Daily Revenue Trajectory")
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.info("Date & Revenue data required for forecast projection.")

        with f_col2:
            st.markdown("#### 📊 Forecast Projections Table")
            if not forecast_df.empty:
                display_f = forecast_df.copy()
                display_f["Predicted_Revenue"] = display_f["Predicted_Revenue"].apply(lambda x: f"${x:,.2f}")
                display_f["Date"] = display_f["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(display_f[["Date", "Predicted_Revenue"]].head(10), use_container_width=True, height=350)

        st.markdown("---")

        # 5. Isolation Forest Anomalies & Time Series Decomposition
        ai_col1, ai_col2 = st.columns([1, 1])
        with ai_col1:
            st.markdown("#### 🚨 Isolation Forest Outlier Detection")
            if not anomalies_df.empty:
                st.warning(f"Detected **{len(anomalies_df)} anomalous transaction(s)** in filtered dataset.")
                rev_col = detected_mappings.get("revenue")
                prod_col = detected_mappings.get("product")
                cols_to_show = [c for c in [detected_mappings.get("date"), prod_col, rev_col, "Anomaly_Score"] if c and c in anomalies_df.columns]
                st.dataframe(anomalies_df[cols_to_show].head(10), use_container_width=True)
            else:
                st.success("✅ No unusual transaction spikes or drops detected in current dataset.")

        with ai_col2:
            fig_decomp = cb.build_time_series_decomposition(filtered_df, detected_mappings)
            st.plotly_chart(fig_decomp, use_container_width=True)

        st.markdown("---")

        # 6. Smart Business Recommendations
        st.markdown("#### 💡 AI Actionable Business Recommendations")
        recommendations = ai.generate_smart_recommendations(filtered_df, detected_mappings)
        rec_cols = st.columns(len(recommendations)) if recommendations else []
        for idx, rec in enumerate(recommendations):
            with rec_cols[idx]:
                st.markdown(
                    f"""
                    <div class="tx-glass-card">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{rec['icon']}</div>
                        <h4 style="margin: 0; color: #F8FAFC;">{rec['title']}</h4>
                        <span style="font-size: 0.75rem; color: #8B5CF6; font-weight: 700;">{rec['category']}</span>
                        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem;">{rec['insight']}</p>
                        <hr style="border-color: rgba(255,255,255,0.06);">
                        <p style="font-size: 0.82rem; color: #CBD5E1;"><b>Action:</b> {rec['action']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =========================================================
    # TAB 3: DATA PROFILING & EXPLORER
    # =========================================================
    with tab3:
        st.markdown("### 🔍 Data Profiling & Column Mapping Engine")

        prof_col1, prof_col2 = st.columns([1, 2])

        profile = generate_data_profile(filtered_df)

        with prof_col1:
            st.markdown("#### 📋 Dataset Metrics")
            st.metric("Total Rows Count", f"{profile['total_rows']:,}")
            st.metric("Total Columns Count", f"{profile['total_columns']}")
            st.metric("Memory Footprint", f"{profile['memory_mb']} MB")

            st.markdown("#### ⚙️ Auto-Detected Column Roles")
            for role, col in detected_mappings.items():
                st.text(f"• {role.capitalize()}: {col if col else 'Not Detected'}")

        with prof_col2:
            st.markdown("#### 📊 Column Quality Profiling")
            st.dataframe(profile["column_profile"], use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Interactive Dataset Explorer")
        search_kw = st.text_input("Filter Raw Dataset Table Rows", key="tab3_table_search")
        
        display_df = filtered_df
        if search_kw:
            mask = np.column_stack([display_df[col].astype(str).str.contains(search_kw, case=False, na=False) for col in display_df.columns])
            display_df = display_df[mask.any(axis=1)]

        st.dataframe(display_df, use_container_width=True, height=400)

    # =========================================================
    # TAB 4: EXPORT & REPORTS
    # =========================================================
    with tab4:
        st.markdown("### 📥 Report Generation & Export Studio")

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.markdown("#### 📄 Executive PDF Report Generation")
            st.write("Generate a formatted executive summary report including KPIs, performance breakdown tables, and official Thiranex branding.")
            
            if st.button("🛠️ Build PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating PDF Report..."):
                    pdf_bytes = exporters.generate_pdf_report(filtered_df, kpis, detected_mappings)
                    st.download_button(
                        label="⬇️ Download Executive PDF Report",
                        data=pdf_bytes,
                        file_name="Thiranex_Executive_Sales_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        with exp_col2:
            st.markdown("#### 📊 Export Filtered Dataset (CSV)")
            st.write("Export the current filtered dataset records as a UTF-8 encoded CSV file.")
            
            csv_bytes = exporters.export_dataframe_to_csv(filtered_df)
            st.download_button(
                label="⬇️ Download Filtered CSV Data",
                data=csv_bytes,
                file_name="Thiranex_Filtered_Sales_Data.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------------------------------------------------------
    # 6. Sidebar FAQ & Footer
    # ---------------------------------------------------------
    ui.render_help_sidebar_faq()
    ui.render_footer()


if __name__ == "__main__":
    main()
