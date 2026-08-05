"""
Module: filters.py
Description: Sidebar controls and dynamic interactive filtering engine for customer dataset.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any


def render_sidebar_filters(df: pd.DataFrame, mappings: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Renders sidebar interactive filters and applies selection masks to the DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Filtered DataFrame and metadata dict.
    """
    st.sidebar.markdown("## ⚙️ Controls & Filters")

    # 1. Segmentation Method Selection
    seg_method = st.sidebar.selectbox(
        "🧠 Segmentation Method",
        ["K-Means Clustering (ML)", "RFM Analysis (Traditional)", "DBSCAN Clustering (Outliers)", "Hierarchical Clustering"],
        key="sb_seg_method",
        help="Select the ML or statistical algorithm to group customers."
    )

    n_clusters = 3
    if seg_method in ["K-Means Clustering (ML)", "Hierarchical Clustering"]:
        n_clusters = st.sidebar.slider(
            "🔢 Number of Clusters (K)",
            min_value=3,
            max_value=6,
            value=3,
            step=1,
            key="sb_n_clusters"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Customer Search & Demographic Filters")

    # 2. Customer Search
    search_query = st.sidebar.text_input(
        "🔎 Search Customer ID",
        value="",
        placeholder="e.g. CUST-0042",
        key="sb_cust_search"
    )

    # 3. Demographic Filters
    # Age Slider
    age_col = mappings.get("age") if mappings.get("age") in df.columns else ("Age" if "Age" in df.columns else None)
    selected_age_range = (18, 70)
    if age_col:
        min_a = int(df[age_col].min()) if not df[age_col].isnull().all() else 18
        max_a = int(df[age_col].max()) if not df[age_col].isnull().all() else 70
        selected_age_range = st.sidebar.slider(
            "👤 Age Range",
            min_value=min_a,
            max_value=max_a,
            value=(min_a, max_a),
            key="sb_age_range"
        )

    # Gender Multi-Select
    gender_col = mappings.get("gender") if mappings.get("gender") in df.columns else ("Gender" if "Gender" in df.columns else None)
    selected_genders = []
    if gender_col:
        all_genders = df[gender_col].dropna().unique().tolist()
        selected_genders = st.sidebar.multiselect(
            "🚻 Gender",
            options=all_genders,
            default=all_genders,
            key="sb_gender_select"
        )

    # Location Multi-Select
    loc_col = mappings.get("location") if mappings.get("location") in df.columns else ("Location" if "Location" in df.columns else None)
    selected_locations = []
    if loc_col:
        all_locations = df[loc_col].dropna().unique().tolist()
        selected_locations = st.sidebar.multiselect(
            "🗺️ Location / Region",
            options=all_locations,
            default=all_locations,
            key="sb_location_select"
        )

    # Income Range Slider
    inc_col = mappings.get("income") if mappings.get("income") in df.columns else ("Income" if "Income" in df.columns else None)
    selected_inc_range = (20000, 150000)
    if inc_col:
        min_i = int(df[inc_col].min()) if not df[inc_col].isnull().all() else 20000
        max_i = int(df[inc_col].max()) if not df[inc_col].isnull().all() else 150000
        selected_inc_range = st.sidebar.slider(
            "💵 Annual Income ($)",
            min_value=min_i,
            max_value=max_i,
            value=(min_i, max_i),
            key="sb_income_range"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛒 Behavioral Filters")

    # Spend Slider
    spend_col = mappings.get("spend") if mappings.get("spend") in df.columns else ("Total Spend ($)" if "Total Spend ($)" in df.columns else None)
    selected_spend_range = (0.0, 10000.0)
    if spend_col:
        min_s = float(df[spend_col].min()) if not df[spend_col].isnull().all() else 0.0
        max_s = float(df[spend_col].max()) if not df[spend_col].isnull().all() else 10000.0
        selected_spend_range = st.sidebar.slider(
            "💰 Total Spend ($)",
            min_value=min_s,
            max_value=max_s,
            value=(min_s, max_s),
            key="sb_spend_range"
        )

    # Frequency Slider
    freq_col = mappings.get("frequency") if mappings.get("frequency") in df.columns else ("Purchase Frequency" if "Purchase Frequency" in df.columns else None)
    selected_freq_range = (1, 50)
    if freq_col:
        min_f = int(df[freq_col].min()) if not df[freq_col].isnull().all() else 1
        max_f = int(df[freq_col].max()) if not df[freq_col].isnull().all() else 50
        selected_freq_range = st.sidebar.slider(
            "📦 Purchase Frequency",
            min_value=min_f,
            max_value=max_f,
            value=(min_f, max_f),
            key="sb_freq_range"
        )

    # Filter Logic Application
    filtered_df = df.copy()

    if search_query:
        id_col = mappings.get("customer_id") if mappings.get("customer_id") in filtered_df.columns else "Customer ID"
        if id_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[id_col].astype(str).str.contains(search_query, case=False, na=False)]

    if age_col and selected_age_range:
        filtered_df = filtered_df[(filtered_df[age_col] >= selected_age_range[0]) & (filtered_df[age_col] <= selected_age_range[1])]

    if gender_col and selected_genders:
        filtered_df = filtered_df[filtered_df[gender_col].isin(selected_genders)]

    if loc_col and selected_locations:
        filtered_df = filtered_df[filtered_df[loc_col].isin(selected_locations)]

    if inc_col and selected_inc_range:
        filtered_df = filtered_df[(filtered_df[inc_col] >= selected_inc_range[0]) & (filtered_df[inc_col] <= selected_inc_range[1])]

    if spend_col and selected_spend_range:
        filtered_df = filtered_df[(filtered_df[spend_col] >= selected_spend_range[0]) & (filtered_df[spend_col] <= selected_spend_range[1])]

    if freq_col and selected_freq_range:
        filtered_df = filtered_df[(filtered_df[freq_col] >= selected_freq_range[0]) & (filtered_df[freq_col] <= selected_freq_range[1])]

    filter_meta = {
        "segmentation_method": seg_method,
        "n_clusters": n_clusters,
        "search_query": search_query,
        "total_records": len(df),
        "filtered_records": len(filtered_df)
    }

    return filtered_df, filter_meta
