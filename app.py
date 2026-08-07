"""
=============================================================================
Thiranex Solutions — Enterprise Data Cleaning & Reporting Automation Platform
Author: Google Deepmind Agentic AI Team
Technology Stack: Streamlit, Pandas, NumPy, Scikit-Learn, OpenPyXL, ReportLab, Plotly
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import json

# Page Configuration
st.set_page_config(
    page_title="Thiranex Solutions — Data Cleaning & Reporting Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# Import Enterprise Modules
from modules.synthetic_messy_data import generate_messy_enterprise_dataset
from modules import data_loader as dl
from modules import data_profiler as dp
from modules import missing_handler as mh
from modules import duplicate_handler as dh
from modules import inconsistency_handler as ih
from modules import outlier_detector as od
from modules import feature_engineer as fe
from modules import data_repair as dr
from modules import pipeline_builder as pb
from modules import report_generator as rg
from modules import scheduler as sch
from modules import chart_builder as cb
from modules import novelty_features as nf
from modules import filters as flt
from modules import exporters as exp
from modules import ui_components as ui

# Inject Custom Glassmorphic Dark Theme Styling
ui.inject_custom_css()

def main():
    """Main Application Execution Routine."""
    ui.render_header_banner()
    ui.render_onboarding_tour()

    # Session State Initialization
    if "lineage_tracker" not in st.session_state:
        st.session_state.lineage_tracker = nf.DataLineageTracker()

    if "df_raw" not in st.session_state:
        st.session_state.df_raw = generate_messy_enterprise_dataset("Customer CRM & Sales", 250)
        st.session_state.lineage_tracker.record_step("Loaded Initial Synthetic Dataset", st.session_state.df_raw)

    if "df_clean" not in st.session_state:
        st.session_state.df_clean = st.session_state.df_raw.copy()

    if "workflow_step" not in st.session_state:
        st.session_state.workflow_step = 1

    if "pipeline_steps" not in st.session_state:
        st.session_state.pipeline_steps = pb.DEFAULT_PIPELINE_PRESET

    # Mode Selector in Sidebar
    app_mode = st.sidebar.radio("🧭 Navigation Mode", ["🧙 Guided 8-Step User Workflow", "📊 Comprehensive Multi-Tab Studio"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Source Selection")
    source_type = st.sidebar.selectbox(
        "Select Data Source",
        [
            "⚡ Synthetic Messy Dataset (Demo)",
            "📤 Single / Batch File Upload",
            "🗂️ Folder Monitoring Watcher",
            "🗄️ Database Integration (SQL / MongoDB)",
            "☁️ Cloud Storage (Google Sheets / S3 / Azure)",
            "🌐 REST API & Web Ingestion"
        ]
    )

    if source_type == "⚡ Synthetic Messy Dataset (Demo)":
        domain = st.sidebar.selectbox("Demo Domain Scenario", ["Customer CRM & Sales", "Financial Transactions", "HR & Employee Records"])
        rows = st.sidebar.slider("Row Count", 50, 1000, 250)
        if st.sidebar.button("🔄 Generate Fresh Dataset", use_container_width=True):
            st.session_state.df_raw = generate_messy_enterprise_dataset(domain, rows)
            st.session_state.df_clean = st.session_state.df_raw.copy()
            st.session_state.lineage_tracker = nf.DataLineageTracker()
            st.session_state.lineage_tracker.record_step(f"Generated {domain}", st.session_state.df_raw)
            st.toast("⚡ Fresh synthetic dataset generated!", icon="✅")

    elif source_type == "📤 Single / Batch File Upload":
        uploaded_files = st.sidebar.file_uploader(
            "Upload Files (CSV, XLSX, XLS, JSON, XML, Parquet, Feather)",
            type=["csv", "xlsx", "xls", "json", "xml", "parquet", "feather"],
            accept_multiple_files=True
        )
        if uploaded_files and st.sidebar.button("📥 Load Uploaded Files", use_container_width=True):
            combined_df, err = dl.load_batch_files(uploaded_files)
            if err:
                st.sidebar.error(err)
            elif combined_df is not None:
                st.session_state.df_raw = combined_df
                st.session_state.df_clean = combined_df.copy()
                st.session_state.lineage_tracker = nf.DataLineageTracker()
                st.session_state.lineage_tracker.record_step("Uploaded Batch Files", combined_df)
                st.toast(f"✅ Loaded {len(combined_df)} records!", icon="🎉")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧹 Quick Operations")
    if st.sidebar.button("🚀 One-Click Auto-Clean", use_container_width=True):
        cleaned_df, logs, impact = pb.execute_pipeline(st.session_state.df_raw, st.session_state.pipeline_steps)
        st.session_state.df_clean = cleaned_df
        st.session_state.lineage_tracker.record_step("Auto-Cleaned Dataset", cleaned_df, impact)
        st.toast(f"🎉 Dataset Cleaned! Quality Score improved by +{impact.get('score_improvement', 0)} points.", icon="✨")

    if st.sidebar.button("↺ Reset Dataset to Original", use_container_width=True):
        st.session_state.df_clean = st.session_state.df_raw.copy()
        st.session_state.lineage_tracker = nf.DataLineageTracker()
        st.session_state.lineage_tracker.record_step("Reset to Raw State", st.session_state.df_raw)
        st.toast("Reset to original raw dataset.", icon="🔄")

    # Generate Profile for Current Working Data
    profile_raw = dp.generate_comprehensive_profile(st.session_state.df_raw)
    profile_clean = dp.generate_comprehensive_profile(st.session_state.df_clean)

    # ---------------------------------------------------------
    # MODE 1: Guided 8-Step User Workflow (Wizard Mode)
    # ---------------------------------------------------------
    if app_mode == "🧙 Guided 8-Step User Workflow":
        ui.render_step_progress(st.session_state.workflow_step)

        # STEP 1: Upload Data or Connect Source
        if st.session_state.workflow_step == 1:
            st.markdown("### 📥 Step 1: Upload Data or Connect Source")
            st.info("Select a data source in the sidebar or preview the current active dataset below:")
            st.dataframe(st.session_state.df_raw.head(20), use_container_width=True)
            col_next, _ = st.columns([1, 4])
            with col_next:
                if st.button("Proceed to Step 2: Auto-Profile ➡️", use_container_width=True):
                    st.session_state.workflow_step = 2
                    st.rerun()

        # STEP 2: Auto-Profiles Data & Quality Score
        elif st.session_state.workflow_step == 2:
            st.markdown("### 📊 Step 2: Automated Quality Profiling & Health Scorecard")
            scorecard = profile_clean.get("scorecard", {})
            st.plotly_chart(cb.build_quality_scorecard_gauges(scorecard), use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Missing Values Heatmap")
                st.plotly_chart(cb.build_missing_values_heatmap(st.session_state.df_raw), use_container_width=True)
            with c2:
                st.markdown("#### Outlier Distribution")
                st.plotly_chart(cb.build_outlier_boxplots(st.session_state.df_raw), use_container_width=True)

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 1", use_container_width=True):
                    st.session_state.workflow_step = 1
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 3: Select Strategy ➡️", use_container_width=True):
                    st.session_state.workflow_step = 3
                    st.rerun()

        # STEP 3: Select Cleaning Strategy
        elif st.session_state.workflow_step == 3:
            st.markdown("### 🧹 Step 3: Select Cleaning Strategy")
            strat_choice = st.radio("Choose Cleaning Approach", [
                "⚡ Auto-Clean (Recommended One-Click Automation)",
                "🛠️ Manual Customization (Specify custom rules per feature)",
                " Visual Pipeline Workflow (Drag-and-Drop sequence)"
            ])

            if strat_choice == "⚡ Auto-Clean (Recommended One-Click Automation)":
                if st.button("🚀 Apply Recommended Auto-Clean", use_container_width=True):
                    cleaned_df, logs, impact = pb.execute_pipeline(st.session_state.df_raw, st.session_state.pipeline_steps)
                    st.session_state.df_clean = cleaned_df
                    st.session_state.lineage_tracker.record_step("Guided Auto-Clean", cleaned_df, impact)
                    st.session_state.workflow_step = 4
                    st.rerun()

            elif strat_choice == "🛠️ Manual Customization (Specify custom rules per feature)":
                c1, c2 = st.columns(2)
                with c1:
                    imp_strat = st.selectbox("Missing Value Strategy", ["Auto-Select", "KNN Imputation", "Mean Imputation", "Median Imputation"])
                    casing_strat = st.selectbox("Text Casing", ["Title Case", "Upper Case", "Lower Case"])
                with c2:
                    out_strat = st.selectbox("Outlier Handling", ["IQR Method", "Z-Score Method", "Isolation Forest"])
                    dupe_strat = st.selectbox("Deduplication", ["Exact Duplicates", "Fuzzy Record Linkage"])

                if st.button("✨ Apply Custom Cleaning Settings", use_container_width=True):
                    df_c, m1 = mh.handle_missing_values(st.session_state.df_raw, strategy=imp_strat)
                    df_c, m2 = dh.handle_duplicates(df_c, method=dupe_strat)
                    df_c, m3 = ih.handle_inconsistencies(df_c, text_case=casing_strat)
                    df_c, m4 = od.detect_and_handle_outliers(df_c, method=out_strat)
                    st.session_state.df_clean = df_c
                    st.session_state.lineage_tracker.record_step("Guided Manual Cleaning", df_c)
                    st.session_state.workflow_step = 4
                    st.rerun()

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 2", use_container_width=True):
                    st.session_state.workflow_step = 2
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 4: Review Comparison ➡️", use_container_width=True):
                    st.session_state.workflow_step = 4
                    st.rerun()

        # STEP 4: Before/After Comparison
        elif st.session_state.workflow_step == 4:
            st.markdown("### 📈 Step 4: Before vs After Cleaning Impact")
            
            imp_metrics = {
                "initial_missing": int(st.session_state.df_raw.isna().sum().sum()),
                "final_missing": int(st.session_state.df_clean.isna().sum().sum()),
                "initial_dupes": int(st.session_state.df_raw.duplicated().sum()),
                "final_dupes": int(st.session_state.df_clean.duplicated().sum()),
                "initial_score": profile_raw["scorecard"]["overall_score"],
                "final_score": profile_clean["scorecard"]["overall_score"]
            }
            st.plotly_chart(cb.build_cleaning_impact_chart(imp_metrics), use_container_width=True)

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 3", use_container_width=True):
                    st.session_state.workflow_step = 3
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 5: Verify Results ➡️", use_container_width=True):
                    st.session_state.workflow_step = 5
                    st.rerun()

        # STEP 5: Verify Results
        elif st.session_state.workflow_step == 5:
            st.markdown("### 🔍 Step 5: Review & Adjust Cleaned Results")
            st.markdown("#### Cleaned Dataset Preview")
            st.dataframe(st.session_state.df_clean, use_container_width=True)
            
            st.markdown("#### Audit Trail Log")
            st.dataframe(st.session_state.lineage_tracker.get_audit_trail_df(), use_container_width=True)

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 4", use_container_width=True):
                    st.session_state.workflow_step = 4
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 6: Select Report ➡️", use_container_width=True):
                    st.session_state.workflow_step = 6
                    st.rerun()

        # STEP 6: Select Report Type & Format
        elif st.session_state.workflow_step == 6:
            st.markdown("### 📄 Step 6: Select Report Type & Output Format")
            c1, c2 = st.columns(2)
            with c1:
                rep_type = st.selectbox("Report Content Type", ["Executive Dashboard (Leadership)", "Technical Cleaning Log", "Comparative Before/After Report"])
            with c2:
                rep_format = st.selectbox("Output File Format", ["PDF", "Excel Workbook", "HTML", "JSON Metadata"])

            st.session_state.selected_rep_format = rep_format

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 5", use_container_width=True):
                    st.session_state.workflow_step = 5
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 7: Export Report ➡️", use_container_width=True):
                    st.session_state.workflow_step = 7
                    st.rerun()

        # STEP 7: Export Report
        elif st.session_state.workflow_step == 7:
            st.markdown("### 💾 Step 7: Generate & Download Report")
            rep_fmt = st.session_state.get("selected_rep_format", "PDF")
            audit_df = st.session_state.lineage_tracker.get_audit_trail_df()
            data_dict_df = nf.generate_smart_metadata_dictionary(st.session_state.df_clean)

            rep_bytes, rep_fname, rep_mime = exp.export_report_bytes(
                st.session_state.df_clean, profile_clean, audit_df, data_dict_df, rep_fmt
            )
            st.download_button(
                label=f"📄 Download Generated {rep_fmt} Report",
                data=rep_bytes,
                file_name=rep_fname,
                mime=rep_mime,
                use_container_width=True
            )

            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Back to Step 6", use_container_width=True):
                    st.session_state.workflow_step = 6
                    st.rerun()
            with col_next:
                if st.button("Proceed to Step 8: Automation ➡️", use_container_width=True):
                    st.session_state.workflow_step = 8
                    st.rerun()

        # STEP 8: Automated Scheduling & Alerts
        elif st.session_state.workflow_step == 8:
            st.markdown("### ⏰ Step 8: Schedule Automated Future Runs")
            c1, c2 = st.columns(2)
            with c1:
                job_title = st.text_input("Scheduled Job Title", value="Daily Customer CRM ETL Clean")
                freq = st.selectbox("Schedule Frequency", ["Daily (00:00 UTC)", "Weekly (Monday 06:00)", "Monthly (1st of Month)"])
            with c2:
                watch_dir = st.text_input("Watch Folder Directory Path", value="./data_inbox")
                st_alert = st.checkbox("Send Stakeholder Email Alerts on Quality Drops", value=True)

            if st.button("⏰ Save & Activate Scheduled Automation Job", use_container_width=True):
                job_meta = sch.create_scheduled_job(job_title, freq, st.session_state.pipeline_steps)
                st.success(f"Job '{job_title}' successfully activated! Job ID: {job_meta['job_id']}")
                st.info("The background engine will auto-process incoming data files and trigger email notifications.")

            if st.button("↺ Start New Workflow Run", use_container_width=True):
                st.session_state.workflow_step = 1
                st.rerun()

    # ---------------------------------------------------------
    # MODE 2: Comprehensive Multi-Tab Studio
    # ---------------------------------------------------------
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📁 Data Overview",
            "📊 Quality Scorecard & Profiling",
            "🧹 Cleaning Studio",
            "⚡ Visual Pipeline Automation",
            "🤖 AI Novelty Suite",
            "📄 Multi-Format Reporting"
        ])

        with tab1:
            st.markdown("### 📋 Active Dataset Summary & Raw Preview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", f"{len(st.session_state.df_clean):,}")
            c2.metric("Total Columns", f"{len(st.session_state.df_clean.columns):,}")
            c3.metric("Memory Usage", f"{profile_clean['structure']['memory_kb']} KB")
            c4.metric("Data Health Score", f"{profile_clean['scorecard']['overall_score']} / 100", delta=f"{profile_clean['scorecard']['badge']}")
            st.dataframe(st.session_state.df_clean, use_container_width=True)

        with tab2:
            st.markdown("### 📊 Enterprise Data Quality Scorecard")
            scorecard = profile_clean.get("scorecard", {})
            st.plotly_chart(cb.build_quality_scorecard_gauges(scorecard), use_container_width=True)
            st.plotly_chart(cb.build_missing_values_heatmap(st.session_state.df_raw, st.session_state.df_clean), use_container_width=True)

        with tab3:
            st.markdown("### 🧹 Intelligent Data Cleaning Studio")
            c1, c2 = st.columns(2)
            with c1:
                imp_mode = st.selectbox("Imputation Strategy", ["Auto-Select", "KNN Imputation", "Random Forest ML Imputation", "Mean Imputation", "Median Imputation"])
            with c2:
                if st.button("Apply Imputation Strategy", use_container_width=True):
                    df_res, meta = mh.handle_missing_values(st.session_state.df_clean, strategy=imp_mode)
                    st.session_state.df_clean = df_res
                    st.session_state.lineage_tracker.record_step(f"Imputed ({imp_mode})", df_res)
                    st.rerun()

        with tab4:
            st.markdown("### ⚡ Visual Pipeline Builder")
            st.dataframe(pd.DataFrame(st.session_state.pipeline_steps), use_container_width=True)
            if st.button("▶️ Execute Visual ETL Pipeline", use_container_width=True):
                cleaned_df, logs, impact = pb.execute_pipeline(st.session_state.df_raw, st.session_state.pipeline_steps)
                st.session_state.df_clean = cleaned_df
                st.session_state.lineage_tracker.record_step("Executed Visual ETL Pipeline", cleaned_df, impact)
                st.success("Visual ETL Pipeline finished!")

        with tab5:
            st.markdown("### 🤖 Novel AI Features Suite")
            nl_q = st.text_input("Talk to Your Data (Plain English)", value="show annual spend > 5000")
            if st.button("Query Data", use_container_width=True):
                res, msg = nf.talk_to_your_data_query(st.session_state.df_clean, nl_q)
                st.info(msg)
                st.dataframe(res, use_container_width=True)

        with tab6:
            st.markdown("### 📄 Multi-Format Reporting & Export Studio")
            r_fmt = st.selectbox("Report Format", ["PDF", "Excel Workbook", "HTML", "JSON Metadata"])
            audit_df = st.session_state.lineage_tracker.get_audit_trail_df()
            data_dict_df = nf.generate_smart_metadata_dictionary(st.session_state.df_clean)
            rep_bytes, rep_fname, rep_mime = exp.export_report_bytes(
                st.session_state.df_clean, profile_clean, audit_df, data_dict_df, r_fmt
            )
            st.download_button(label=f"📄 Download {r_fmt} Report", data=rep_bytes, file_name=rep_fname, mime=rep_mime, use_container_width=True)

    ui.render_footer()

if __name__ == "__main__":
    main()
