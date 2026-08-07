"""
=============================================================================
Thiranex Solutions — Intelligent Data Cleaning & Transformation Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

def clean_missing_values(df: pd.DataFrame, strategy: str = "Auto-Select", custom_cols: List[str] = None, threshold_pct: float = 50.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes or drops missing values based on specified strategy:
    Strategies: 'Drop Rows', 'Drop Columns', 'Mean Imputation', 'Median Imputation',
                'Mode Imputation', 'Forward Fill', 'Backward Fill', 'Linear Interpolation',
                'KNN Imputation', 'Random Forest ML Imputation', 'Auto-Select'
    """
    df_clean = df.copy()
    initial_nulls = int(df_clean.isna().sum().sum())
    cols_to_process = custom_cols if custom_cols else df_clean.columns.tolist()
    
    if strategy == "Drop Columns (> Threshold)":
        drop_cols = [c for c in cols_to_process if (df_clean[c].isna().sum() / len(df_clean)) * 100.0 > threshold_pct]
        df_clean = df_clean.drop(columns=drop_cols)
        final_nulls = int(df_clean.isna().sum().sum())
        return df_clean, {"action": f"Dropped {len(drop_cols)} columns with missingness > {threshold_pct}%", "initial_nulls": initial_nulls, "final_nulls": final_nulls}
        
    elif strategy == "Drop Rows":
        df_clean = df_clean.dropna(subset=cols_to_process)
        final_nulls = int(df_clean.isna().sum().sum())
        return df_clean, {"action": f"Dropped rows containing missing values in {cols_to_process}", "initial_nulls": initial_nulls, "final_nulls": final_nulls}
        
    elif strategy in ["Mean Imputation", "Median Imputation", "Mode Imputation", "Auto-Select"]:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.intersection(cols_to_process)
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.intersection(cols_to_process)
        
        for c in num_cols:
            if df_clean[c].isna().sum() > 0:
                if strategy == "Median Imputation" or (strategy == "Auto-Select" and abs(df_clean[c].skew()) > 1.0):
                    val = df_clean[c].median()
                else:
                    val = df_clean[c].mean()
                df_clean[c] = df_clean[c].fillna(val)
                
        for c in cat_cols:
            if df_clean[c].isna().sum() > 0:
                mode_val = df_clean[c].mode()[0] if not df_clean[c].mode().empty else "Unknown"
                df_clean[c] = df_clean[c].fillna(mode_val)
                
    elif strategy == "Forward Fill":
        df_clean[cols_to_process] = df_clean[cols_to_process].ffill().bfill()
        
    elif strategy == "Backward Fill":
        df_clean[cols_to_process] = df_clean[cols_to_process].bfill().ffill()
        
    elif strategy == "Linear Interpolation":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.intersection(cols_to_process)
        df_clean[num_cols] = df_clean[num_cols].interpolate(method="linear").bfill().ffill()
        
    elif strategy == "KNN Imputation":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.intersection(cols_to_process)
        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            
    elif strategy == "Random Forest ML Imputation":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.intersection(cols_to_process)
        for c in num_cols:
            if df_clean[c].isna().sum() > 0:
                other_num = [col for col in num_cols if col != c and df_clean[col].isna().sum() == 0]
                if other_num:
                    train_data = df_clean[df_clean[c].notna()]
                    test_data = df_clean[df_clean[c].isna()]
                    rf = RandomForestRegressor(n_estimators=30, random_state=42)
                    rf.fit(train_data[other_num], train_data[c])
                    preds = rf.predict(test_data[other_num])
                    df_clean.loc[df_clean[c].isna(), c] = preds
                else:
                    df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    final_nulls = int(df_clean.isna().sum().sum())
    return df_clean, {
        "strategy": strategy,
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "imputed_cells": initial_nulls - final_nulls
    }

def handle_duplicates(df: pd.DataFrame, method: str = "Exact Duplicates", keep: str = "first", similarity_threshold: float = 0.85, text_col: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Purges exact or fuzzy duplicates from the DataFrame.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    if method == "Exact Duplicates":
        keep_arg = keep if keep in ["first", "last"] else False
        df_clean = df_clean.drop_duplicates(keep=keep_arg).reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        return df_clean, {"method": "Exact Deduplication", "initial_rows": initial_rows, "final_rows": len(df_clean), "removed_records": removed}
        
    elif method == "Fuzzy Record Linkage" and text_col and text_col in df_clean.columns:
        # Detect text similarity using SequenceMatcher
        series = df_clean[text_col].astype(str).tolist()
        to_drop = set()
        n = len(series)
        # Compare strings efficiently
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, min(i + 50, n)):
                if j in to_drop:
                    continue
                ratio = SequenceMatcher(None, series[i].lower(), series[j].lower()).ratio()
                if ratio >= similarity_threshold:
                    to_drop.add(j)
                    
        df_clean = df_clean.drop(index=list(to_drop)).reset_index(drop=True)
        removed = len(to_drop)
        return df_clean, {"method": f"Fuzzy Linkage ({text_col} >= {similarity_threshold})", "initial_rows": initial_rows, "final_rows": len(df_clean), "removed_records": removed}

    return df_clean, {"method": "None", "initial_rows": initial_rows, "final_rows": initial_rows, "removed_records": 0}

def standardize_data(df: pd.DataFrame, text_case: str = "Title Case", date_cols: List[str] = None, phone_cols: List[str] = None, convert_units: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes date formats, text casing, phone formatting, and performs unit conversion.
    """
    df_clean = df.copy()
    changes_log = []
    
    # 1. Standardize Text Casing
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        if text_case == "Upper Case":
            df_clean[c] = df_clean[c].astype(str).str.upper()
        elif text_case == "Lower Case":
            df_clean[c] = df_clean[c].astype(str).str.lower()
        elif text_case == "Title Case":
            df_clean[c] = df_clean[c].astype(str).str.title().str.strip()
    if cat_cols.any():
        changes_log.append(f"Standardized text casing to {text_case}")

    # 2. Date Parsing & ISO Standardization
    if date_cols:
        for dc in date_cols:
            if dc in df_clean.columns:
                df_clean[dc] = pd.to_datetime(df_clean[dc], errors="coerce").dt.strftime("%Y-%m-%d")
                changes_log.append(f"Parsed & standardized dates in '{dc}' to YYYY-MM-DD")

    # 3. Phone Number Standardization (Format as (XXX) XXX-XXXX)
    if phone_cols:
        for pc in phone_cols:
            if pc in df_clean.columns:
                def _fmt_phone(val):
                    digits = re.sub(r"\D", "", str(val))
                    if len(digits) == 10:
                        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                    elif len(digits) == 11 and digits.startswith("1"):
                        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
                    return val
                df_clean[pc] = df_clean[pc].apply(_fmt_phone)
                changes_log.append(f"Standardized phone numbers in '{pc}'")

    # 4. Unit Conversion (e.g. 'kg to lbs' or 'lbs to kg')
    if convert_units:
        for col, mode in convert_units.items():
            if col in df_clean.columns:
                def _convert(val):
                    try:
                        s_val = str(val).lower()
                        num = float(re.findall(r"[-+]?\d*\.\d+|\d+", s_val)[0])
                        if mode == "kg -> lbs" and "kg" in s_val:
                            return f"{round(num * 2.20462, 1)} lbs"
                        elif mode == "lbs -> kg" and "lbs" in s_val:
                            return f"{round(num / 2.20462, 1)} kg"
                        return val
                    except Exception:
                        return val
                df_clean[col] = df_clean[col].apply(_convert)
                changes_log.append(f"Converted units in '{col}' ({mode})")

    return df_clean, {"changes": changes_log}

def detect_and_handle_outliers(df: pd.DataFrame, method: str = "IQR Method", action: str = "Capping (Winsorizing)", target_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects and handles numeric outliers via IQR, Z-score, Isolation Forest, or DBSCAN.
    Actions: 'Capping (Winsorizing)', 'Remove Outliers', 'Log Transformation'
    """
    df_clean = df.copy()
    num_cols = target_cols if target_cols else df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df_clean, {"outliers_detected": 0, "action": "None"}

    total_outliers = 0
    
    if method == "IQR Method":
        for c in num_cols:
            s = df_clean[c].dropna()
            if len(s) > 4:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                mask = (df_clean[c] < lower_bound) | (df_clean[c] > upper_bound)
                total_outliers += int(mask.sum())
                
                if action == "Capping (Winsorizing)":
                    df_clean[c] = np.clip(df_clean[c], lower_bound, upper_bound)
                elif action == "Remove Outliers":
                    df_clean = df_clean[~mask]
                elif action == "Log Transformation":
                    min_val = df_clean[c].min()
                    shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0
                    df_clean[c] = np.log1p(df_clean[c] + shift)

    elif method == "Z-Score Method":
        for c in num_cols:
            s = df_clean[c].dropna()
            if len(s) > 4 and s.std() > 0:
                z = np.abs((df_clean[c] - s.mean()) / s.std())
                mask = z > 3.0
                total_outliers += int(mask.sum())
                if action == "Capping (Winsorizing)":
                    df_clean.loc[df_clean[c] > s.mean() + 3 * s.std(), c] = s.mean() + 3 * s.std()
                    df_clean.loc[df_clean[c] < s.mean() - 3 * s.std(), c] = s.mean() - 3 * s.std()
                elif action == "Remove Outliers":
                    df_clean = df_clean[~mask]

    elif method == "Isolation Forest":
        valid_num = df_clean[num_cols].dropna()
        if len(valid_num) > 10:
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(valid_num)
            outlier_idx = valid_num.index[preds == -1]
            total_outliers = len(outlier_idx)
            if action == "Remove Outliers":
                df_clean = df_clean.drop(index=outlier_idx)

    return df_clean, {
        "method": method,
        "action": action,
        "outliers_detected": total_outliers,
        "remaining_rows": len(df_clean)
    }

def optimize_memory_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimizes numeric dtypes downcasting int64/float64 to int32/float32 and converts high cardinality text to categories.
    """
    df_clean = df.copy()
    start_mem = df_clean.memory_usage(deep=True).sum() / 1024.0
    
    # Downcast numerics
    for col in df_clean.select_dtypes(include=["int64", "int32"]).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast="integer")
        
    for col in df_clean.select_dtypes(include=["float64"]).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast="float")

    # Convert text to categorical if unique ratio < 0.3
    for col in df_clean.select_dtypes(include=["object"]).columns:
        if df_clean[col].nunique() / max(len(df_clean), 1) < 0.3:
            df_clean[col] = df_clean[col].astype("category")

    end_mem = df_clean.memory_usage(deep=True).sum() / 1024.0
    saved_pct = round(((start_mem - end_mem) / max(start_mem, 1)) * 100.0, 1)

    return df_clean, {
        "start_memory_kb": round(start_mem, 2),
        "end_memory_kb": round(end_mem, 2),
        "saved_pct": saved_pct
    }

def normalize_and_encode(df: pd.DataFrame, norm_method: str = "None", encoding_method: str = "None", target_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies feature scaling (Min-Max, StandardScaler, RobustScaler) and categorical encoding (One-Hot, Label Encoding).
    """
    df_clean = df.copy()
    log = []
    
    # 1. Normalization
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

    # 2. Categorical Encoding
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
