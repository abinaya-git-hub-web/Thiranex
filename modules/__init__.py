"""
=============================================================================
Thiranex Solutions Data Cleaning & Reporting Suite — Modular Engine Package
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

from .synthetic_messy_data import generate_messy_enterprise_dataset
from .data_loader import load_file, load_batch_files, scan_folder, connect_sql_database, connect_nosql_mongodb, fetch_cloud_storage, ingest_rest_api
from .data_profiler import generate_comprehensive_profile, calculate_scorecard
from .missing_handler import handle_missing_values
from .duplicate_handler import handle_duplicates
from .inconsistency_handler import handle_inconsistencies
from .outlier_detector import detect_and_handle_outliers
from .feature_engineer import normalize_and_encode_features, engineer_date_and_text_features
from .data_repair import ml_predictive_imputation
from .pipeline_builder import execute_pipeline, export_pipeline_preset, import_pipeline_preset, DEFAULT_PIPELINE_PRESET
from .report_generator import generate_excel_report, generate_pdf_report, generate_html_report
from .scheduler import create_scheduled_job, monitor_directory_for_new_files
from .chart_builder import *
from .filters import render_interactive_filters
from .exporters import export_dataframe_bytes, export_report_bytes
from .ui_components import inject_custom_css, render_header_banner, render_kpi_card, render_onboarding_tour, render_footer
