"""
=============================================================================
Thiranex Solutions — Enterprise Predictive Analytics Application
Author: Google Deepmind Agentic AI Team
Technology Stack: Streamlit, Scikit-learn, Statsmodels, Prophet, Plotly, ReportLab
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Page Configuration
st.set_page_config(
    page_title="Thiranex Solutions — Enterprise Predictive Analytics Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import Modular Engine Components according to Thiranex Architecture
from modules.synthetic_data import generate_synthetic_timeseries
from modules.data_loader import load_uploaded_file, detect_column_types, generate_data_profile
from modules.preprocessor import (
    clean_time_series_data, perform_stationarity_tests,
    engineer_time_series_features, split_time_series
)
from modules.model_selector import run_automl_suite, fit_predict_holt_winters
from modules.model_evaluator import (
    calculate_forecasting_metrics, analyze_residuals, compute_prediction_intervals
)
from modules.scenario_analyzer import simulate_what_if_scenario
from modules.insight_generator import generate_ai_executive_summary
from modules import chart_builder as cb
from modules import novelty_features as nf
from modules.filters import render_sidebar_controls
from modules import exporters
from modules import ui_components as ui

warnings.filterwarnings("ignore")

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
        [
            "⚡ Synthetic Demo Data (3+ Years Daily)",
            "📤 Upload Custom Dataset (CSV, Excel, JSON)"
        ],
        key="sb_data_source"
    )

    df_raw = None

    if data_option == "📤 Upload Custom Dataset (CSV, Excel, JSON)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Time-Series File",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )
        if uploaded_file is not None:
            with st.spinner("⏳ Parsing uploaded time-series dataset..."):
                df_raw, upload_error = load_uploaded_file(uploaded_file)
            if upload_error:
                st.error(f"❌ {upload_error}")
                st.stop()
        else:
            st.info("👆 Please upload a time-series dataset to proceed, or switch to Synthetic Demo Data.")
            st.stop()
    else:
        scenario = st.sidebar.selectbox(
            "Demo Scenario",
            ["Sales Data", "Revenue Data", "Website Traffic"],
            index=0
        )
        with st.spinner("⚡ Generating 3+ years realistic time-series dataset with seasonal & holiday dynamics..."):
            df_raw = generate_synthetic_timeseries(scenario=scenario)

    # ---------------------------------------------------------
    # 2. Smart Column Detection & Preprocessing Engine
    # ---------------------------------------------------------
    detected_mappings = detect_column_types(df_raw)
    ctrl = render_sidebar_controls(df_raw, detected_mappings)

    date_col = ctrl["date_col"]
    target_col = ctrl["target_col"]

    # Automated Cleaning
    df_clean, clean_meta = clean_time_series_data(
        df_raw, date_col=date_col, target_col=target_col,
        impute_method=ctrl["impute_method"],
        outlier_method=ctrl["outlier_method"],
        smooth_method=ctrl["smooth_method"]
    )

    # Feature Engineering
    numeric_features = [c for c in df_clean.columns if c not in [date_col, target_col] and pd.api.types.is_numeric_dtype(df_clean[c])]
    df_engineered = engineer_time_series_features(
        df_clean, date_col=date_col, target_col=target_col,
        numeric_features=numeric_features, max_lags=ctrl["max_lags"],
        include_calendar=ctrl["include_calendar"], include_fourier=ctrl["include_fourier"]
    )

    # Train-Test Split
    train_df, test_df = split_time_series(df_engineered, test_ratio=ctrl["test_split_ratio"])
    feature_cols = [c for c in train_df.columns if c not in [date_col, target_col] and pd.api.types.is_numeric_dtype(train_df[c])]

    # ---------------------------------------------------------
    # 3. Model Execution & AutoML Engine
    # ---------------------------------------------------------
    with st.spinner("🧠 Executing Predictive AI Model Suite & AutoML Leaderboard..."):
        automl_res = run_automl_suite(
            train_df, test_df, date_col=date_col,
            target_col=target_col, feature_cols=feature_cols
        )

    leaderboard = automl_res["leaderboard"]
    best_model_name = automl_res["best_model_name"]
    recommender_info = nf.recommend_optimal_model(leaderboard)

    # Selected Model Prediction Logic
    sel_model_str = ctrl["selected_model"]

    if "AutoML" in sel_model_str:
        active_model_name = best_model_name
    elif "Ensemble" in sel_model_str:
        active_model_name = "Ensemble (Top 3 Weighted)"
    else:
        active_model_name = sel_model_str

    active_data = leaderboard.get(active_model_name, leaderboard.get(best_model_name))
    y_test_actual = test_df[target_col].values
    y_test_pred = active_data["forecast"]

    # Compute Metrics & Confidence Bands
    metrics = calculate_forecasting_metrics(y_test_actual, y_test_pred, train_df[target_col].values)
    res_analysis = analyze_residuals(y_test_actual, y_test_pred)
    lower_bound, upper_bound = compute_prediction_intervals(y_test_pred, res_analysis["std"], ctrl["confidence_level"])

    # Future Forecast Horizon Simulation
    future_dates = pd.date_range(start=df_clean[date_col].max() + pd.Timedelta(days=1), periods=ctrl["forecast_horizon"], freq="D")
    future_forecast, _, _ = fit_predict_holt_winters(df_clean[target_col], ctrl["forecast_horizon"])
    fut_lower, fut_upper = compute_prediction_intervals(future_forecast, res_analysis["std"], ctrl["confidence_level"])

    # Novelty Elements Calculations
    scenario_fc, scenario_summary = simulate_what_if_scenario(
        y_test_pred,
        marketing_change_pct=ctrl["marketing_change_pct"],
        price_change_pct=ctrl["price_change_pct"],
        competitor_impact_pct=ctrl["competitor_impact_pct"]
    )
    risk_info = nf.generate_risk_advised_forecast(y_test_pred, lower_bound, upper_bound)
    anomalies_info = nf.monitor_prediction_anomalies(test_df[date_col], y_test_actual, y_test_pred)

    ai_executive_summary = generate_ai_executive_summary(
        target_col, active_model_name, metrics,
        float(np.sum(y_test_pred)), float(np.sum(y_test_actual)),
        risk_info, anomalies_info["anomaly_count"]
    )

    # ---------------------------------------------------------
    # 4. Header Banner & Onboarding Tour
    # ---------------------------------------------------------
    ui.render_header_banner()
    ui.render_onboarding_tour()
    ui.render_help_sidebar()

    # ---------------------------------------------------------
    # 5. Application Tabs (6 Business-Ready Modules)
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Executive Summary & AutoML",
        "📈 Forecast Studio & What-If Simulator",
        "🔍 Residual Diagnostics & Stationarity",
        "🧠 Explainable AI & Feature Drivers",
        "🌐 Multi-Variate & Budget Variance",
        "📥 Export Studio & REST API"
    ])

    # =========================================================
    # TAB 1: EXECUTIVE SUMMARY & AUTOML LEADERBOARD
    # =========================================================
    with tab1:
        st.markdown("#### 🚀 Executive KPI Performance Cards")
        k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)

        with k_col1:
            ui.render_kpi_card("Total Observations", f"{len(df_clean):,}", icon="📊")
        with k_col2:
            ui.render_kpi_card("Forecasted Sum", f"${np.sum(y_test_pred):,.2f}", icon="💰")
        with k_col3:
            ui.render_kpi_card("Active Model", active_model_name, icon="🤖")
        with k_col4:
            ui.render_kpi_card("RMSE Error", f"{metrics['rmse']:,.2f}", icon="🎯")
        with k_col5:
            ui.render_kpi_card("MAE Error", f"{metrics['mae']:,.2f}", icon="📉")
        with k_col6:
            ui.render_kpi_card("MAPE Accuracy", f"{metrics['mape']:.2f}%", icon="⭐")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1)); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
                {ai_executive_summary}
            </div>
            """,
            unsafe_allow_html=True
        )

        r1_col1, r1_col2 = st.columns([1, 1])
        with r1_col1:
            st.markdown("##### 🏆 AutoML Model Performance Leaderboard")
            lead_df = pd.DataFrame([
                {"Model": m, "RMSE": v["rmse"], "MAE": v["mae"], "MAPE (%)": v["mape"]}
                for m, v in leaderboard.items()
            ]).sort_values(by="RMSE").reset_index(drop=True)
            st.dataframe(lead_df.style.highlight_min(axis=0, color="#1E3A8A"), use_container_width=True)

            st.info(f"💡 **Recommender Engine Verdict**: {recommender_info['reason']}")

        with r1_col2:
            fig_comp = cb.build_model_comparison_chart(leaderboard)
            cb.render_plotly_chart(fig_comp, key="chart_leaderboard_comp")

        st.markdown("<br>", unsafe_allow_html=True)

        r2_col1, r2_col2 = st.columns([1, 1])
        with r2_col1:
            fig_overlay = cb.build_actual_vs_predicted_chart(
                test_df[date_col], y_test_actual, y_test_pred,
                lower_bound, upper_bound,
                title=f"Actual vs Predicted ({active_model_name})"
            )
            cb.render_plotly_chart(fig_overlay, key="chart_tab1_overlay")
        with r2_col2:
            fig_horizon = cb.build_forecast_horizon_chart(
                df_clean[date_col], df_clean[target_col].values,
                future_dates, future_forecast,
                fut_lower, fut_upper
            )
            cb.render_plotly_chart(fig_horizon, key="chart_tab1_horizon")

    # =========================================================
    # TAB 2: FORECAST STUDIO & WHAT-IF SIMULATOR
    # =========================================================
    with tab2:
        st.markdown("#### 📈 Interactive What-If Business Scenario Simulator")

        scen_col1, scen_col2 = st.columns([1, 2])
        with scen_col1:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 1rem;">
                    <h4 style="color: #10B981; margin-top: 0;">🔮 What-If Scenario Impact</h4>
                    <p><b>Baseline Forecast Total</b>: ${scenario_summary['base_total']:,.2f}</p>
                    <p><b>Simulated Forecast Total</b>: ${scenario_summary['simulated_total']:,.2f}</p>
                    <p><b>Net Financial Delta</b>: <span style="color: {'#10B981' if scenario_summary['net_delta'] >= 0 else '#EF4444'}; font-weight: 700;">${scenario_summary['net_delta']:,.2f} ({scenario_summary['pct_change']}%)</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color: #3B82F6; margin-top: 0;">🛡️ Risk-Advised Decision Support</h4>
                    <p><b>Risk Rating</b>: {risk_info['risk_level']}</p>
                    <p><b>P10 Conservative Total</b>: ${risk_info['p10_conservative_total']:,.2f}</p>
                    <p><b>P50 Expected Total</b>: ${risk_info['p50_expected_total']:,.2f}</p>
                    <p><b>P90 Optimistic Total</b>: ${risk_info['p90_optimistic_total']:,.2f}</p>
                    <p style="font-size: 0.85rem; color: #94A3B8;"><b>Strategic Advice</b>: {risk_info['advice']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with scen_col2:
            fig_sim = cb.build_scenario_simulation_chart(test_df[date_col], y_test_pred, scenario_fc)
            cb.render_plotly_chart(fig_sim, key="chart_what_if_sim")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🚨 Prediction Anomaly Monitor (Real-Time Deviations)")
        if anomalies_info["anomalies"]:
            st.dataframe(pd.DataFrame(anomalies_info["anomalies"]), use_container_width=True)
        else:
            st.success("✅ No prediction anomalies detected exceeding tolerance threshold.")

    # =========================================================
    # TAB 3: RESIDUAL DIAGNOSTICS & STATIONARITY
    # =========================================================
    with tab3:
        st.markdown("#### 🔍 Time-Series Diagnostics & Residual Analysis")

        stationarity_res = perform_stationarity_tests(df_clean[target_col])
        st.info(f"📊 **Stationarity Assessment**: {stationarity_res['verdict']} (ADF p-value: {stationarity_res['adf']['p_value']:.4f}, KPSS p-value: {stationarity_res['kpss']['p_value']:.4f})")

        fig_res_dash = cb.build_residual_analysis_dashboard(res_analysis["residuals"])
        cb.render_plotly_chart(fig_res_dash, key="chart_res_dash")

        st.markdown("<br>", unsafe_allow_html=True)

        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            fig_rf = cb.build_residual_vs_fitted_chart(y_test_pred, res_analysis["residuals"])
            cb.render_plotly_chart(fig_rf, key="chart_rf")

            fig_acf = cb.build_acf_pacf_chart(res_analysis["acf"], res_analysis["pacf"])
            cb.render_plotly_chart(fig_acf, key="chart_acf")

        with d_col2:
            fig_qq = cb.build_qq_plot(res_analysis["qq_theoretical"], res_analysis["qq_sample"])
            cb.render_plotly_chart(fig_qq, key="chart_qq")

            fig_decomp = cb.build_seasonal_decomposition_chart(df_clean[target_col])
            cb.render_plotly_chart(fig_decomp, key="chart_decomp")

    # =========================================================
    # TAB 4: EXPLAINABLE AI & FEATURE DRIVERS
    # =========================================================
    with tab4:
        st.markdown("#### 🧠 Explainable AI Dashboard & External Impact Analysis")

        feature_drivers = nf.generate_feature_importance_breakdown(active_model_name, feature_cols, train_df, target_col)

        e_col1, e_col2 = st.columns([1, 1])
        with e_col1:
            fig_fi = cb.build_feature_importance_chart(feature_drivers)
            cb.render_plotly_chart(fig_fi, key="chart_fi")
        with e_col2:
            fig_shap = cb.build_shap_explainability_chart(feature_drivers)
            cb.render_plotly_chart(fig_shap, key="chart_shap")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### 🌐 External Variable Impact Analyzer")
        ext_impacts = nf.analyze_external_variable_impact(df_clean, target_col, numeric_features)
        if ext_impacts:
            st.dataframe(pd.DataFrame(ext_impacts), use_container_width=True)
        else:
            st.info("No external regressor variables present in dataset.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### 🔄 Adaptive Learning & Model Drift Detector")
        drift_info = nf.detect_model_drift(y_test_actual, y_test_pred, baseline_mape=metrics["mape"])
        st.markdown(
            f"""
            <div class="glass-card">
                <h5 style="margin:0; color: {'#EF4444' if drift_info['retrain_needed'] else '#10B981'};">{drift_info['drift_status']}</h5>
                <p style="margin: 0.4rem 0 0 0; color: #CBD5E1;">
                    Current Test MAPE: <b>{drift_info['current_mape']}%</b> | Baseline MAPE: <b>{drift_info['baseline_mape']}%</b> | Performance Shift: <b>{drift_info['degradation_pct']}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================================================
    # TAB 5: MULTI-VARIATE & BUDGET VARIANCE
    # =========================================================
    with tab5:
        st.markdown("#### 🌐 Multi-Variate Forecasting & Budget Target Variance")

        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            budget_res = nf.calculate_budget_variance(y_test_pred, ctrl["budget_target"])
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color: {budget_res['color']}; margin-top: 0;">🎯 Budget Target Variance Analysis</h4>
                    <p><b>Forecast Total</b>: ${budget_res['forecast_total']:,.2f}</p>
                    <p><b>Budget Target</b>: ${budget_res['budget_target']:,.2f}</p>
                    <p><b>Variance Amount</b>: <span style="color: {budget_res['color']}; font-weight: 700;">${budget_res['variance_amount']:,.2f} ({budget_res['variance_pct']}%)</span></p>
                    <p><b>Variance Status</b>: {budget_res['status']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with b_col2:
            fig_roll = cb.build_rolling_forecast_chart(y_test_actual, y_test_pred)
            cb.render_plotly_chart(fig_roll, key="chart_roll_val")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### 🔮 Multi-Target Forecast Predictions")
        multi_targets = [c for c in ["Sales_Units", "Revenue_USD", "Units_Sold", "Profit_USD", target_col] if c in df_clean.columns]
        if len(multi_targets) > 1:
            multi_res = nf.run_multi_target_forecast(train_df, test_df, list(set(multi_targets)))
            multi_df = pd.DataFrame([
                {"Target Column": k, "Forecast Total": v["total_predicted"], "Mean / Step": v["mean_predicted"], "Test RMSE": v["rmse"]}
                for k, v in multi_res.items()
            ])
            st.dataframe(multi_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_heat = cb.build_prediction_error_heatmap(y_test_actual, y_test_pred)
        cb.render_plotly_chart(fig_heat, key="chart_tab5_heatmap")

    # =========================================================
    # TAB 6: EXPORT STUDIO & REST API
    # =========================================================
    with tab6:
        st.markdown("#### 📥 Enterprise Export Studio & Integration APIs")

        ex_col1, ex_col2, ex_col3 = st.columns(3)

        with ex_col1:
            st.markdown("##### 📄 Download CSV Predictions")
            csv_data = exporters.generate_forecast_csv(
                test_df[date_col], y_test_actual, y_test_pred,
                lower_bound, upper_bound, active_model_name
            )
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"thiranex_forecast_{target_col}.csv",
                mime="text/csv"
            )

        with ex_col2:
            st.markdown("##### 📑 Executive PDF Report")
            if st.button("⚡ Generate ReportLab Executive PDF"):
                with st.spinner("Generating Executive PDF Briefing..."):
                    pdf_data = exporters.generate_executive_pdf_report(
                        target_col, active_model_name, metrics,
                        leaderboard, risk_info, ai_executive_summary
                    )
                st.download_button(
                    label="📥 Download Executive PDF Report",
                    data=pdf_data,
                    file_name=f"Thiranex_Executive_Predictive_Report.pdf",
                    mime="application/pdf"
                )

        with ex_col3:
            st.markdown("##### 🌐 REST API Response Endpoint")
            api_json = exporters.generate_forecast_json_api(
                target_col, active_model_name, metrics, y_test_pred, risk_info
            )
            st.download_button(
                label="📥 Download REST API JSON Payload",
                data=api_json,
                file_name=f"forecast_api_payload.json",
                mime="application/json"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### 💻 Programmatic REST API JSON Preview")
        st.code(api_json, language="json")

    # Render Footer
    ui.render_footer()


if __name__ == "__main__":
    main()
