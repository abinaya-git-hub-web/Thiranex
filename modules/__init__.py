"""
Thiranex Solutions Customer Segmentation Suite — Modules Package
"""

from .data_loader import generate_sample_data, load_uploaded_file
from .preprocessor import detect_column_types, generate_data_profile, extract_and_scale_features
from .rfm_analyzer import calculate_rfm_scores
from .kmeans_cluster import run_kmeans, compute_elbow_and_silhouette
from .dbscan_cluster import run_dbscan
from .hierarchical_cluster import run_hierarchical
from .persona_builder import generate_segment_personas, generate_automated_narrative
from .ai_predictor import predict_customer_churn, predict_customer_clv, generate_next_best_offer, analyze_purchase_patterns
from .chart_builder import *
from .filters import render_sidebar_filters
from .exporters import generate_pdf_personas_report, export_dataframe_to_csv
from .ui_components import inject_custom_css, render_header_banner, render_kpi_card, render_persona_card, render_onboarding_tour, render_footer

__all__ = [
    "generate_sample_data", "load_uploaded_file",
    "detect_column_types", "generate_data_profile", "extract_and_scale_features",
    "calculate_rfm_scores",
    "run_kmeans", "compute_elbow_and_silhouette",
    "run_dbscan",
    "run_hierarchical",
    "generate_segment_personas", "generate_automated_narrative",
    "predict_customer_churn", "predict_customer_clv", "generate_next_best_offer", "analyze_purchase_patterns",
    "render_sidebar_filters",
    "generate_pdf_personas_report", "export_dataframe_to_csv",
    "inject_custom_css", "render_header_banner", "render_kpi_card", "render_persona_card", "render_onboarding_tour", "render_footer"
]
