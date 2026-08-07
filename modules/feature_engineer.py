"""
=============================================================================
Thiranex Solutions — Feature Engineering & Normalization Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

def normalize_and_encode_features(df: pd.DataFrame, norm_method: str = "None", encoding_method: str = "None", target_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies feature scaling (Min-Max, StandardScaler, RobustScaler) and categorical encoding (One-Hot, Label Encoding).
    """
    df_clean = df.copy()
    log = []

    num_cols = target_cols if target_cols else df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if norm_method != "None" and num_cols:
        if norm_method == "Min-Max Scaler":
            scaler = MinMaxScaler()
        elif norm_method == "Z-Score (StandardScaler)":
            scaler = StandardScaler()
        elif norm_method == "Robust Scaler":
            scaler = RobustScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
        log.append(f"Applied {norm_method} to {len(num_cols)} numerical features.")

    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    if encoding_method != "None" and cat_cols:
        if encoding_method == "One-Hot Encoding":
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
            log.append(f"One-Hot encoded {len(cat_cols)} categorical columns.")
        elif encoding_method == "Label Encoding":
            le = LabelEncoder()
            for c in cat_cols:
                df_clean[c] = le.fit_transform(df_clean[c].astype(str))
            log.append(f"Label encoded {len(cat_cols)} categorical columns.")

    return df_clean, {"log": log}

def engineer_date_and_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automated feature creation: extracts date parts (year, month, day, dayofweek) and text lengths.
    """
    df_clean = df.copy()
    
    # Date parts
    for c in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[c]):
            df_clean[f"{c}_Year"] = df_clean[c].dt.year
            df_clean[f"{c}_Month"] = df_clean[c].dt.month
            df_clean[f"{c}_Day"] = df_clean[c].dt.day
            df_clean[f"{c}_DayOfWeek"] = df_clean[c].dt.dayofweek

    # Text stats
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        if "id" not in c.lower():
            df_clean[f"{c}_Len"] = df_clean[c].astype(str).str.len()

    return df_clean
