"""
Module: filters.py
Description: Advanced filtering controls, date range presets, multi-select dropdowns, 
             range sliders, search box, filter reset, and active filter counter badge.
"""

import datetime
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional, List, Any


def render_sidebar_filters(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Renders sidebar filter controls and applies selected criteria to filter the dataset.

    Args:
        df (pd.DataFrame): Input preprocessed DataFrame.
        mappings (Dict[str, Optional[str]]): Column role mappings.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Filtered DataFrame and metadata dict (active filter count, active filters summary).
    """
    st.sidebar.markdown("### 🎛️ Filter Control Panel")

    if df.empty:
        return df, {"count": 0, "summary": []}

    date_col = mappings.get("date")
    cat_col = mappings.get("category")
    region_col = mappings.get("region")
    rep_col = mappings.get("salesperson")
    pay_col = mappings.get("payment_method")
    prod_col = mappings.get("product")
    price_col = mappings.get("price")
    qty_col = mappings.get("quantity")

    # Reset Filters Button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    active_filters_count = 0
    active_filters_summary = []

    filtered_df = df.copy()

    # 1. Date Range Presets & Picker
    if date_col and date_col in filtered_df.columns:
        st.sidebar.markdown("#### 📅 Date Filter")
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        min_date = filtered_df[date_col].min().date()
        max_date = filtered_df[date_col].max().date()

        preset = st.sidebar.selectbox(
            "Date Range Preset",
            ["All Time", "Last 7 Days", "This Month", "Last Quarter", "Year-to-Date", "Custom"],
            key="sb_date_preset"
        )

        today = max_date  # Reference anchor date

        if preset == "Last 7 Days":
            start_d = max(min_date, today - datetime.timedelta(days=7))
            end_d = today
            active_filters_count += 1
            active_filters_summary.append("Date: Last 7 Days")
        elif preset == "This Month":
            start_d = datetime.date(today.year, today.month, 1)
            end_d = today
            active_filters_count += 1
            active_filters_summary.append("Date: This Month")
        elif preset == "Last Quarter":
            start_d = max(min_date, today - datetime.timedelta(days=90))
            end_d = today
            active_filters_count += 1
            active_filters_summary.append("Date: Last Quarter")
        elif preset == "Year-to-Date":
            start_d = datetime.date(today.year, 1, 1)
            end_d = today
            active_filters_count += 1
            active_filters_summary.append("Date: YTD")
        elif preset == "Custom":
            selected_dates = st.sidebar.date_input(
                "Custom Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="sb_date_custom"
            )
            if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                start_d, end_d = selected_dates
                if (start_d, end_d) != (min_date, max_date):
                    active_filters_count += 1
                    active_filters_summary.append(f"Date: {start_d} to {end_d}")
            else:
                start_d, end_d = min_date, max_date
        else:
            start_d, end_d = min_date, max_date

        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_d) & 
            (filtered_df[date_col].dt.date <= end_d)
        ]

    st.sidebar.markdown("---")

    # 2. Multi-Select Dropdowns
    # Category
    if cat_col and cat_col in filtered_df.columns:
        cat_options = sorted(list(filtered_df[cat_col].dropna().unique()))
        selected_cats = st.sidebar.multiselect("Category", cat_options, key="sb_cats")
        if selected_cats:
            filtered_df = filtered_df[filtered_df[cat_col].isin(selected_cats)]
            active_filters_count += 1
            active_filters_summary.append(f"Category: {len(selected_cats)} selected")

    # Region
    if region_col and region_col in filtered_df.columns:
        region_options = sorted(list(filtered_df[region_col].dropna().unique()))
        selected_regions = st.sidebar.multiselect("Region", region_options, key="sb_regions")
        if selected_regions:
            filtered_df = filtered_df[filtered_df[region_col].isin(selected_regions)]
            active_filters_count += 1
            active_filters_summary.append(f"Region: {len(selected_regions)} selected")

    # Salesperson
    if rep_col and rep_col in filtered_df.columns:
        rep_options = sorted(list(filtered_df[rep_col].dropna().unique()))
        selected_reps = st.sidebar.multiselect("Salesperson", rep_options, key="sb_reps")
        if selected_reps:
            filtered_df = filtered_df[filtered_df[rep_col].isin(selected_reps)]
            active_filters_count += 1
            active_filters_summary.append(f"Salesperson: {len(selected_reps)} selected")

    # Payment Method
    if pay_col and pay_col in filtered_df.columns:
        pay_options = sorted(list(filtered_df[pay_col].dropna().unique()))
        selected_pays = st.sidebar.multiselect("Payment Method", pay_options, key="sb_pays")
        if selected_pays:
            filtered_df = filtered_df[filtered_df[pay_col].isin(selected_pays)]
            active_filters_count += 1
            active_filters_summary.append(f"Payment Method: {len(selected_pays)} selected")

    st.sidebar.markdown("---")

    # 3. Dynamic Range Sliders
    if price_col and price_col in filtered_df.columns:
        min_p = float(df[price_col].min())
        max_p = float(df[price_col].max())
        if min_p < max_p:
            price_range = st.sidebar.slider(
                "Unit Price Range ($)",
                min_value=min_p,
                max_value=max_p,
                value=(min_p, max_p),
                key="sb_price_slider"
            )
            if price_range != (min_p, max_p):
                filtered_df = filtered_df[
                    (filtered_df[price_col] >= price_range[0]) & 
                    (filtered_df[price_col] <= price_range[1])
                ]
                active_filters_count += 1
                active_filters_summary.append(f"Price: ${price_range[0]:.0f}-${price_range[1]:.0f}")

    if qty_col and qty_col in filtered_df.columns:
        min_q = int(df[qty_col].min())
        max_q = int(df[qty_col].max())
        if min_q < max_q:
            qty_range = st.sidebar.slider(
                "Order Quantity Range",
                min_value=min_q,
                max_value=max_q,
                value=(min_q, max_q),
                key="sb_qty_slider"
            )
            if qty_range != (min_q, max_q):
                filtered_df = filtered_df[
                    (filtered_df[qty_col] >= qty_range[0]) & 
                    (filtered_df[qty_col] <= qty_range[1])
                ]
                active_filters_count += 1
                active_filters_summary.append(f"Qty: {qty_range[0]}-{qty_range[1]}")

    st.sidebar.markdown("---")

    # 4. Product Text Search Box
    search_query = st.sidebar.text_input("🔍 Search Product", key="sb_prod_search")
    if search_query and prod_col and prod_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[prod_col].astype(str).str.contains(search_query, case=False, na=False)]
        active_filters_count += 1
        active_filters_summary.append(f"Search: '{search_query}'")

    meta = {
        "count": active_filters_count,
        "summary": active_filters_summary
    }

    return filtered_df, meta
