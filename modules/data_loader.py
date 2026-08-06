"""
=============================================================================
Thiranex Solutions — Smart Data Loader & Profiling Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import io
import json
from typing import Dict, Any, Tuple, Optional, List


def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parses CSV, Excel (.xlsx, .xls), and JSON file uploads safely into a DataFrame.
    """
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".json"):
            # Attempt standard pandas json read or list of dicts
            content = uploaded_file.read().decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for records orientation or single dict
                if "data" in data and isinstance(data["data"], list):
                    df = pd.DataFrame(data["data"])
                else:
                    df = pd.DataFrame([data])
            else:
                return None, "Invalid JSON structure. Expected array of objects or key-value dictionary."
        else:
            return None, f"Unsupported file extension: {uploaded_file.name}"

        if df.empty:
            return None, "The uploaded dataset is empty."

        return df, None
    except Exception as e:
        return None, f"Error parsing file '{uploaded_file.name}': {str(e)}"


def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically detects time-series date/timestamp column, target variable column,
    numerical feature columns (regressors), categorical columns, and time series frequency.
    """
    cols = df.columns.tolist()

    date_candidates = ["date", "timestamp", "period", "time", "datetime", "day", "month", "year"]
    detected_date_col = None

    # 1. Detect Date Column
    for c in cols:
        c_lower = str(c).lower().strip()
        if any(keyword in c_lower for keyword in date_candidates):
            detected_date_col = c
            break

    if not detected_date_col:
        # Fallback: test if any column can be parsed as datetime
        for c in cols:
            if df[c].dtype == "object" or "datetime" in str(df[c].dtype).lower():
                try:
                    parsed = pd.to_datetime(df[c].dropna().head(20), errors="coerce")
                    if parsed.notna().sum() > 15:
                        detected_date_col = c
                        break
                except Exception:
                    pass

    if not detected_date_col and len(cols) > 0:
        detected_date_col = cols[0]

    # 2. Detect Target Column
    target_keywords = ["sales", "revenue", "orders", "units", "traffic", "demand", "value", "y", "target"]
    detected_target_col = None

    for c in cols:
        if c == detected_date_col:
            continue
        c_lower = str(c).lower().strip()
        if any(keyword in c_lower for keyword in target_keywords):
            detected_target_col = c
            break

    if not detected_target_col:
        # Fallback: first numeric non-date column
        numeric_cols = [c for c in cols if c != detected_date_col and pd.api.types.is_numeric_dtype(df[c])]
        detected_target_col = numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else cols[0])

    # 3. Detect Regressor / Feature Columns
    numeric_feature_cols = [
        c for c in cols
        if c not in [detected_date_col, detected_target_col] and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_feature_cols = [
        c for c in cols
        if c not in [detected_date_col, detected_target_col] and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # 4. Detect Frequency
    detected_freq = "Daily"
    if detected_date_col in df.columns:
        try:
            dates = pd.to_datetime(df[detected_date_col], errors="coerce").dropna().sort_values()
            if len(dates) > 5:
                diffs = dates.diff().dropna()
                median_days = diffs.dt.days.median()
                if median_days <= 1:
                    detected_freq = "Daily (D)"
                elif 6 <= median_days <= 8:
                    detected_freq = "Weekly (W)"
                elif 25 <= median_days <= 32:
                    detected_freq = "Monthly (M)"
                elif 85 <= median_days <= 95:
                    detected_freq = "Quarterly (Q)"
                elif median_days >= 350:
                    detected_freq = "Yearly (Y)"
        except Exception:
            detected_freq = "Daily (D)"

    return {
        "date_col": detected_date_col,
        "target_col": detected_target_col,
        "numeric_features": numeric_feature_cols,
        "categorical_features": categorical_feature_cols,
        "frequency": detected_freq
    }


def generate_data_profile(df: pd.DataFrame, mappings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes dataset profiling stats: missing values, data types, summary statistics,
    and simple seasonality indicators.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_total = int(df.isnull().sum().sum())
    missing_pct = round((missing_total / (total_rows * total_cols)) * 100, 2) if total_rows > 0 else 0

    target_col = mappings.get("target_col")
    target_stats = {}
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        s = df[target_col].dropna()
        target_stats = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis())
        }

    date_col = mappings.get("date_col")
    date_range_str = "N/A"
    if date_col and date_col in df.columns:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not parsed_dates.empty:
            date_range_str = f"{parsed_dates.min().strftime('%Y-%m-%d')} to {parsed_dates.max().strftime('%Y-%m-%d')}"

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "date_range": date_range_str,
        "frequency": mappings.get("frequency", "Daily (D)"),
        "target_stats": target_stats
    }
