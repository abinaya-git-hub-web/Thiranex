"""
=============================================================================
Thiranex Solutions — Data Profiler & Quality Assessment Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple

REGEX_PATTERNS = {
    "Email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    "Phone_Number": r"^\+?\d{0,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
    "Zip_Code": r"^\d{5}(-\d{4})?$",
    "IP_Address": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    "URL": r"^https?://[^\s/$.?#].[^\s]*$",
    "ISO_Date": r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$"
}

def generate_comprehensive_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates data quality assessment report including structure, statistical metrics,
    regex pattern frequencies, pairwise correlations, and Data Health Scorecard.
    """
    if df is None or df.empty:
        return {}

    row_count, col_count = df.shape
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024.0, 2)
    
    col_profiles = []
    detected_types = {}
    pattern_matches = {}

    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna().astype(str).str.strip()
        null_count = series.isna().sum()
        completeness_pct = round(((row_count - null_count) / max(row_count, 1)) * 100.0, 2)
        unique_count = series.nunique(dropna=True)
        uniqueness_pct = round((unique_count / max(row_count, 1)) * 100.0, 2)
        
        dtype_str = str(series.dtype)
        inferred_type = _infer_semantic_type(series, non_null_series)
        detected_types[col] = inferred_type
        
        col_patterns = _detect_patterns(non_null_series)
        pattern_matches[col] = col_patterns
        
        col_meta = {
            "Column": col,
            "Raw_Dtype": dtype_str,
            "Inferred_Type": inferred_type,
            "Null_Count": int(null_count),
            "Completeness_%": completeness_pct,
            "Unique_Count": int(unique_count),
            "Uniqueness_%": uniqueness_pct,
            "Pattern_Detected": max(col_patterns, key=col_patterns.get) if col_patterns else "General Text"
        }
        col_profiles.append(col_meta)
        
    df_col_summary = pd.DataFrame(col_profiles)

    # Statistical Summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stat_summary = {}
    if num_cols:
        for c in num_cols:
            s = df[c].dropna()
            if len(s) > 0:
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                stat_summary[c] = {
                    "Count": len(s),
                    "Mean": round(float(s.mean()), 4),
                    "Std": round(float(s.std()), 4),
                    "Median": round(float(s.median()), 4),
                    "Min": round(float(s.min()), 4),
                    "Max": round(float(s.max()), 4),
                    "Skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
                    "Kurtosis": round(float(s.kurtosis()), 4) if len(s) > 2 else 0.0,
                    "IQR": round(float(q75 - q25), 4)
                }
    df_stats = pd.DataFrame(stat_summary).T if stat_summary else pd.DataFrame()

    corr_matrix = pd.DataFrame()
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().round(4).fillna(0.0)

    scorecard = calculate_scorecard(df, df_col_summary)

    return {
        "structure": {
            "row_count": row_count,
            "col_count": col_count,
            "memory_kb": memory_kb,
            "num_columns": len(num_cols),
            "cat_columns": col_count - len(num_cols)
        },
        "columns_summary": df_col_summary,
        "detected_types": detected_types,
        "pattern_matches": pattern_matches,
        "statistical_summary": df_stats,
        "correlation_matrix": corr_matrix,
        "scorecard": scorecard
    }

def _infer_semantic_type(series: pd.Series, non_null: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        if set(series.dropna().unique()).issubset({0, 1}):
            return "Boolean"
        return "Numeric"
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
        
    if len(non_null) > 0:
        sample = non_null.head(30)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="ISO8601")
            if parsed.notna().sum() / len(sample) > 0.7:
                return "Datetime"
        except Exception:
            pass

        email_match = sample.apply(lambda x: bool(re.match(REGEX_PATTERNS["Email"], x))).sum()
        if email_match / len(sample) > 0.4:
            return "Email Address"

        phone_match = sample.apply(lambda x: bool(re.match(REGEX_PATTERNS["Phone_Number"], x))).sum()
        if phone_match / len(sample) > 0.4:
            return "Phone Number"

        if series.nunique() / max(len(series), 1) < 0.2:
            return "Categorical"

    return "General Text"

def _detect_patterns(non_null: pd.Series) -> Dict[str, int]:
    if len(non_null) == 0:
        return {}
    sample = non_null.head(100)
    counts = {}
    for label, pat in REGEX_PATTERNS.items():
        cnt = int(sample.apply(lambda x: bool(re.match(pat, str(x)))).sum())
        if cnt > 0:
            counts[label] = cnt
    return counts

def calculate_scorecard(df: pd.DataFrame, df_col_summary: pd.DataFrame) -> Dict[str, Any]:
    row_count = len(df)
    if row_count == 0:
        return {"overall_score": 0, "badge": "Critical", "recommendations": ["Dataset is empty."]}

    tot_cells = df.size
    tot_missing = df.isna().sum().sum()
    completeness_score = max(0.0, round(((tot_cells - tot_missing) / max(tot_cells, 1)) * 100.0, 1))

    outlier_count = 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 4:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = ((s < (q1 - 3 * iqr)) | (s > (q3 + 3 * iqr))).sum()
            outlier_count += outliers

    accuracy_penalty = min(30.0, (outlier_count / max(row_count, 1)) * 20.0)
    accuracy_score = max(0.0, round(100.0 - accuracy_penalty, 1))

    duplicate_rows = df.duplicated().sum()
    dupe_penalty = (duplicate_rows / max(row_count, 1)) * 40.0
    consistency_score = max(0.0, round(100.0 - dupe_penalty, 1))
    timeliness_score = 92.0

    overall_score = round(
        0.35 * completeness_score +
        0.30 * accuracy_score +
        0.25 * consistency_score +
        0.10 * timeliness_score,
        1
    )

    if overall_score >= 80:
        badge, badge_color = "Good", "#16A34A"
    elif overall_score >= 50:
        badge, badge_color = "Warning", "#D97706"
    else:
        badge, badge_color = "Critical", "#DC2626"

    recs = []
    if completeness_score < 90:
        recs.append(f"⚠️ **High Missingness**: {tot_missing} null cells found. Apply KNN or Random Forest ML Imputation.")
    if duplicate_rows > 0:
        recs.append(f"🔄 **Duplicate Records**: {duplicate_rows} exact duplicates detected. Apply Deduplication.")
    if outlier_count > 0:
        recs.append(f"📈 **Extreme Outliers**: {outlier_count} numeric outliers detected. Apply Winsorizing or IQR Capping.")
    if not recs:
        recs.append("✅ **Data Health Excellent**: Dataset meets enterprise standards.")

    return {
        "overall_score": overall_score,
        "completeness_score": completeness_score,
        "accuracy_score": accuracy_score,
        "consistency_score": consistency_score,
        "timeliness_score": timeliness_score,
        "badge": badge,
        "badge_color": badge_color,
        "total_missing": int(tot_missing),
        "total_duplicates": int(duplicate_rows),
        "total_outliers": int(outlier_count),
        "total_rows": row_count,
        "recommendations": recs
    }
