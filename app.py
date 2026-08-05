"""
=============================================================================
Thiranex Solutions — Enterprise Customer Segmentation & Intelligence Suite
Author: Google Deepmind Agentic AI Team
Technology Stack: Streamlit, Plotly, Pandas, Scikit-Learn, SciPy, ReportLab, Matplotlib, Seaborn
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import Modular Engine Components according to Thiranex Architecture
from modules.data_loader import generate_sample_data, load_uploaded_file
from modules.preprocessor import detect_column_types, generate_data_profile, extract_and_scale_features
from modules import rfm_analyzer as rfm
from modules import kmeans_cluster as km
from modules import dbscan_cluster as db
from modules import hierarchical_cluster as hc
from modules import persona_builder as ps
from modules import ai_predictor as ai
from modules import chart_builder as cb
from modules.filters import render_sidebar_filters
from modules import exporters
from modules import ui_components as ui
from modules.kpi_calculator import calculate_kpis

# Streamlit Page Configuration
st.set_page_config(
    page_title="Thiranex Solutions — Enterprise Customer Analytics Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Glassmorphic Dark Theme Styling
ui.inject_custom_css()


def main():
    """Main Application Execution Routine."""

    # ---------------------------------------------------------
    # 1. Sidebar Data Import & Engine Setup
    # ---------------------------------------------------------
    st.sidebar.markdown("### 📁 Data Source Selection")

    data_option = st.sidebar.radio(
        "Select Data Source",
        ["Use Synthetic Demo Data (1,000 customers)", "Upload Custom Dataset (CSV, Excel, JSON)"],
        key="sb_data_source"
    )

    df_raw = None
    upload_error = None

    if data_option == "Upload Custom Dataset (CSV, Excel, JSON)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Customer File",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )
        if uploaded_file is not None:
            with st.spinner("⏳ Parsing uploaded customer dataset..."):
                df_raw, upload_error = load_uploaded_file(uploaded_file)
            if upload_error:
                st.error(f"❌ {upload_error}")
                st.stop()
        else:
            st.info("👆 Please upload a customer dataset to proceed, or switch to Synthetic Demo Data.")
            st.stop()
    else:
        with st.spinner("⚡ Generating 1,000 synthetic customer profiles with realistic behavior..."):
            df_raw = generate_sample_data(num_rows=1000)

    # ---------------------------------------------------------
    # 2. Smart Column Detection & RFM Preprocessing
    # ---------------------------------------------------------
    detected_mappings = detect_column_types(df_raw)
    df_rfm = rfm.calculate_rfm_scores(df_raw, detected_mappings)

    # ---------------------------------------------------------
    # 3. Dynamic Sidebar Filters
    # ---------------------------------------------------------
    filtered_df, filter_meta = render_sidebar_filters(df_rfm, detected_mappings)

    # ---------------------------------------------------------
    # 4. Clustering Execution Engine
    # ---------------------------------------------------------
    seg_method = filter_meta["segmentation_method"]
    n_clusters = filter_meta.get("n_clusters", 3)

    # Feature extraction & scaling
    scaled_matrix, feature_names, feature_df = extract_and_scale_features(filtered_df, detected_mappings)

    # Apply selected segmentation algorithm
    cluster_labels = []
    active_seg_column = "Active_Segment"

    if seg_method == "K-Means Clustering (ML)":
        kmeans_res = km.run_kmeans(scaled_matrix, n_clusters=n_clusters)
        cluster_labels = [f"K-Means Cluster {i+1}" for i in kmeans_res["labels"]]
    elif seg_method == "RFM Analysis (Traditional)":
        cluster_labels = filtered_df["RFM_Segment"].tolist()
    elif seg_method == "DBSCAN Clustering (Outliers)":
        dbscan_res = db.run_dbscan(scaled_matrix, eps=0.8, min_samples=5)
        cluster_labels = [f"DBSCAN Group {lbl}" if lbl != -1 else "Noise / Outliers" for lbl in dbscan_res["labels"]]
    elif seg_method == "Hierarchical Clustering":
        hier_res = hc.run_hierarchical(scaled_matrix, n_clusters=n_clusters)
        cluster_labels = [f"Hierarchical Group {i+1}" for i in hier_res["labels"]]

    filtered_df[active_seg_column] = cluster_labels

    # Generate Personas & Narrative
    personas = ps.generate_segment_personas(filtered_df, active_seg_column, detected_mappings)
    narrative_summary = ps.generate_automated_narrative(filtered_df, active_seg_column, personas)

    # Calculate KPIs
    kpis = calculate_kpis(filtered_df, detected_mappings)

    # ---------------------------------------------------------
    # 5. Header Banner & Onboarding Guide
    # ---------------------------------------------------------
    ui.render_header_banner(filter_meta)
    ui.render_onboarding_tour()

    # ---------------------------------------------------------
    # 6. Main Application 5 Business-Ready Output Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary Dashboard",
        "👤 Segment Details & Persona Cards",
        "🎯 Marketing Recommendations Report",
        "🚨 Churn Risk & CLV Report",
        "📥 Clustering Diagnostics & Export Studio"
    ])

    # =========================================================
    # TAB 1: EXECUTIVE SUMMARY DASHBOARD (LEADERSHIP VIEW)
    # =========================================================
    with tab1:
        st.markdown("#### 🚀 Executive KPI Overview (Leadership View)")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            ui.render_kpi_card("Total Customers", f"{kpis['total_customers']:,}", icon="👥")
        with col2:
            ui.render_kpi_card("Total Revenue", f"${kpis['total_spend']:,.0f}", icon="💰")
        with col3:
            ui.render_kpi_card("Avg Spend / Client", f"${kpis['avg_spend']:,.2f}", icon="💳")
        with col4:
            ui.render_kpi_card("Avg Order Value", f"${kpis['avg_aov']:,.2f}", icon="🛒")
        with col5:
            ui.render_kpi_card("Avg Purchase Freq", f"{kpis['avg_frequency']} / yr", icon="📦")
        with col6:
            ui.render_kpi_card("Satisfaction Score", f"{kpis['avg_satisfaction']} / 5.0", icon="⭐")

        st.markdown("<br>", unsafe_allow_html=True)

        # AI Executive Narrative Banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1)); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <h4 style="color: #A855F7; margin: 0 0 0.5rem 0;">✨ AI Executive Narrative & Leadership Summary</h4>
                <div style="color: #CBD5E1; font-size: 0.92rem;">{narrative_summary}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visualizations Grid Row 1: Donut & Radar
        r1_col1, r1_col2 = st.columns([1, 1])
        with r1_col1:
            fig_donut = cb.build_segment_distribution_chart(filtered_df, active_seg_column)
            cb.render_plotly_chart(fig_donut, key="chart_donut")
        with r1_col2:
            fig_radar = cb.build_radar_comparison_chart(filtered_df, active_seg_column, detected_mappings)
            cb.render_plotly_chart(fig_radar, key="chart_radar")

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations Grid Row 2: Demographic Heatmap & Spend Box Plot
        r2_col1, r2_col2 = st.columns([1, 1])
        with r2_col1:
            fig_heat = cb.build_demographic_heatmap(filtered_df, active_seg_column, detected_mappings)
            cb.render_plotly_chart(fig_heat, key="chart_heat")
        with r2_col2:
            fig_box = cb.build_spending_boxplot(filtered_df, active_seg_column, detected_mappings)
            cb.render_plotly_chart(fig_box, key="chart_box")

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations Grid Row 3: 3D RFM & Geographic Map
        r3_col1, r3_col2 = st.columns([1, 1])
        with r3_col1:
            fig_3d = cb.build_rfm_3d_scatter(filtered_df, rfm_col=active_seg_column)
            cb.render_plotly_chart(fig_3d, key="chart_3d")
        with r3_col2:
            fig_geo = cb.build_geographic_chart(filtered_df, active_seg_column, detected_mappings)
            cb.render_plotly_chart(fig_geo, key="chart_geo")

    # =========================================================
    # TAB 2: SEGMENT DETAILS & PERSONA CARDS PAGE
    # =========================================================
    with tab2:
        st.markdown("### 👤 Deep Dive: Segment Details & Customer Personas")

        if personas:
            selected_persona_name = st.selectbox(
                "Select Segment Persona to Inspect Details",
                options=[p["creative_name"] for p in personas],
                key="tab2_persona_select"
            )

            target_persona = next((p for p in personas if p["creative_name"] == selected_persona_name), personas[0])
            ui.render_persona_card(target_persona)

            st.markdown("---")
            st.markdown("### ⚔️ Side-by-Side Persona Comparison View")
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                p1_name = st.selectbox("Select Segment A", [p["creative_name"] for p in personas], index=0, key="sb_comp_p1")
                p1 = next((p for p in personas if p["creative_name"] == p1_name), personas[0])
                ui.render_persona_card(p1)

            with comp_col2:
                default_idx = 1 if len(personas) > 1 else 0
                p2_name = st.selectbox("Select Segment B", [p["creative_name"] for p in personas], index=default_idx, key="sb_comp_p2")
                p2 = next((p for p in personas if p["creative_name"] == p2_name), personas[default_idx])
                ui.render_persona_card(p2)
        else:
            st.warning("No segment personas generated for current view.")

    # =========================================================
    # TAB 3: MARKETING RECOMMENDATIONS REPORT
    # =========================================================
    with tab3:
        st.markdown("### 🎯 Strategic Marketing Recommendations & Next Best Offers")

        nbo_list = ai.generate_next_best_offer(filtered_df, active_seg_column)
        nbo_df = pd.DataFrame(nbo_list)
        st.dataframe(nbo_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💡 Actionable Segment Targeting Strategy Cards")

        if personas:
            rec_cols = st.columns(len(personas)) if len(personas) <= 4 else st.columns(4)
            for idx, p in enumerate(personas[:4]):
                with rec_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid {p['theme_color']}; border-radius: 12px; padding: 1.2rem; height: 100%;">
                            <h4 style="color: {p['theme_color']}; margin: 0 0 0.5rem 0;">{p['creative_name']}</h4>
                            <p style="font-size: 0.85rem; color: #CBD5E1;"><b>Target Channel:</b> Direct VIP Email & In-App</p>
                            <p style="font-size: 0.85rem; color: #94A3B8;"><b>Strategy:</b> {p['strategy']}</p>
                            <hr style="border-color: rgba(255,255,255,0.08);">
                            <p style="font-size: 0.85rem; color: #10B981;"><b>Recommended Offer:</b> {p['recommended_offers']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # =========================================================
    # TAB 4: CHURN RISK & CLV REPORT
    # =========================================================
    with tab4:
        st.markdown("### 🚨 Churn Risk Report & Customer Lifetime Value (CLV) Projections")

        # 1. Churn Prediction (Random Forest)
        st.markdown("#### ⚠️ At-Risk Customer Analysis (Random Forest Model)")
        df_churn, seg_churn_rates, churn_drivers = ai.predict_customer_churn(filtered_df, detected_mappings)

        ch_col1, ch_col2 = st.columns([1, 1])
        with ch_col1:
            high_risk_cnt = int((df_churn["Churn_Probability"] > 65.0).sum())
            st.warning(f"Detected **{high_risk_cnt} customers at high risk of churn** (>65% probability).")
            fig_churn = cb.build_churn_driver_chart(churn_drivers)
            cb.render_plotly_chart(fig_churn, key="chart_churn")

        with ch_col2:
            st.markdown("##### 📈 Average Churn Risk per Segment")
            churn_rates_df = pd.DataFrame(list(seg_churn_rates.items()), columns=["Segment", "Avg Churn Rate (%)"])
            st.dataframe(churn_rates_df, use_container_width=True)

        st.markdown("---")

        # 2. Customer Lifetime Value (CLV) Prediction
        st.markdown("#### 💎 Customer Lifetime Value (CLV) Tier Projections")
        df_clv, clv_stats = ai.predict_customer_clv(filtered_df, detected_mappings)

        clv_c1, clv_c2 = st.columns([1, 1])
        with clv_c1:
            st.metric("Total Portfolio Future CLV", f"${clv_stats['total_portfolio_clv']:,.2f}")
            st.metric("Average Historic CLV", f"${clv_stats['avg_historic_clv']:,.2f}")
            st.metric("Predicted 2-Year Avg CLV", f"${clv_stats['avg_predicted_clv']:,.2f}")

        with clv_c2:
            fig_clv = cb.build_clv_distribution_chart(df_clv)
            cb.render_plotly_chart(fig_clv, key="chart_clv")

    # =========================================================
    # TAB 5: CLUSTERING DIAGNOSTICS & EXPORT STUDIO
    # =========================================================
    with tab5:
        st.markdown("### 📥 Model Diagnostics & Report Export Studio")

        diag_c1, diag_c2 = st.columns([1, 1])

        with diag_c1:
            st.markdown("#### 📐 K-Means Optimal K Evaluation")
            elbow_data = km.compute_elbow_and_silhouette(scaled_matrix)
            fig_elbow = cb.build_elbow_chart(elbow_data["k_values"], elbow_data["inertias"])
            cb.render_plotly_chart(fig_elbow, key="chart_elbow")

            fig_sil = cb.build_silhouette_chart(elbow_data["k_values"], elbow_data["silhouette_scores"])
            cb.render_plotly_chart(fig_sil, key="chart_sil")

        with diag_c2:
            st.markdown("#### 🌳 Agglomerative Dendrogram Tree")
            hier_res = hc.run_hierarchical(scaled_matrix, n_clusters=3)
            fig_dendro = cb.build_dendrogram_chart(hier_res["linkage_matrix"])
            cb.render_plotly_chart(fig_dendro, key="chart_dendro")

        st.markdown("---")
        st.markdown("#### 🔀 Customer Lifecycle Journey Flow")
        fig_sankey = cb.build_lifecycle_sankey(filtered_df, active_seg_column)
        cb.render_plotly_chart(fig_sankey, key="chart_sankey")

        st.markdown("---")

        prof_col1, prof_col2 = st.columns([1, 2])
        profile = generate_data_profile(filtered_df)

        with prof_col1:
            st.markdown("#### 📋 Dataset Profile")
            st.metric("Total Customer Rows", f"{profile['total_rows']:,}")
            st.metric("Total Attributes Count", f"{profile['total_columns']}")
            st.metric("Memory Footprint", f"{profile['memory_mb']} MB")

        with prof_col2:
            st.markdown("#### 📊 Column Quality Profiling")
            st.dataframe(profile["column_profile"], use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Segmented Dataset Explorer")
        st.dataframe(filtered_df, use_container_width=True, height=300)

        st.markdown("---")

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.markdown("#### 📄 Executive PDF Persona Report")
            st.write("Download an executive PDF report featuring personas, marketing strategies, and key metrics.")
            if st.button("🛠️ Generate Executive PDF Report", type="primary", use_container_width=True):
                with st.spinner("Building PDF document..."):
                    pdf_bytes = exporters.generate_pdf_personas_report(filtered_df, personas, seg_method)
                    st.download_button(
                        label="⬇️ Download Executive PDF Report",
                        data=pdf_bytes,
                        file_name="Thiranex_Customer_Segmentation_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        with exp_col2:
            st.markdown("#### 📊 Export Segmented Data (CSV)")
            st.write("Export full customer records with cluster labels, RFM scores, churn risk, and CLV tiers.")
            csv_bytes = exporters.export_dataframe_to_csv(filtered_df)
            st.download_button(
                label="⬇️ Download Segmented CSV Dataset",
                data=csv_bytes,
                file_name="Thiranex_Segmented_Customers.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------------------------------------------------------
    # 7. Application Footer
    # ---------------------------------------------------------
    ui.render_footer()


if __name__ == "__main__":
    main()
