"""
=============================================================================
Thiranex Solutions Predictive Analytics Suite — Modular Engine Package
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

from .data_loader import load_uploaded_file, detect_column_types, generate_data_profile
from .synthetic_data import generate_synthetic_timeseries
from .preprocessor import clean_time_series_data, perform_stationarity_tests, engineer_time_series_features, split_time_series
from .arima_model import fit_predict_sarima
from .prophet_model import fit_predict_prophet
from .regression_models import fit_predict_regression_model
from .neural_network import fit_predict_neural_network
from .model_selector import run_automl_suite, fit_predict_holt_winters
from .model_evaluator import calculate_forecasting_metrics, analyze_residuals, compute_prediction_intervals
from .chart_builder import *
from .scenario_analyzer import simulate_what_if_scenario
from .insight_generator import generate_ai_executive_summary
from .novelty_features import *
from .filters import render_sidebar_controls
from .exporters import generate_forecast_csv, generate_forecast_json_api, generate_executive_pdf_report
from .ui_components import inject_custom_css, render_header_banner, render_kpi_card, render_onboarding_tour, render_footer
