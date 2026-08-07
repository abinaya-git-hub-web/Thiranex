"""
=============================================================================
Thiranex Solutions — Interactive Filters Component Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def render_interactive_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renders dynamic column filtering controls for data exploration.
    """
    if df is None or df.empty:
        return df

    df_filtered = df.copy()
    st.sidebar.markdown("### 🎛️ Interactive Dataset Filters")

    # Select columns to filter on
    filter_cols = st.sidebar.multiselect("Select Filter Columns", df_filtered.columns.tolist()[:5])

    for col in filter_cols:
        if pd.api.types.is_numeric_dtype(df_filtered[col]):
            min_val, max_val = float(df_filtered[col].min()), float(df_filtered[col].max())
            selected_range = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            df_filtered = df_filtered[(df_filtered[col] >= selected_range[0]) & (df_filtered[col] <= selected_range[1])]

        elif pd.api.types.is_string_dtype(df_filtered[col]) or pd.api.types.is_categorical_dtype(df_filtered[col]):
            unique_vals = df_filtered[col].dropna().unique().tolist()[:20]
            selected_cats = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            if selected_cats:
                df_filtered = df_filtered[df_filtered[col].isin(selected_cats)]

    return df_filtered
