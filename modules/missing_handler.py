"""
=============================================================================
Thiranex Solutions — Missing Value Handling Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.impute import KNNImputer

def handle_missing_values(df: pd.DataFrame, strategy: str = "Auto-Select", custom_cols: List[str] = None, threshold_pct: float = 50.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes or drops missing values based on specified strategy.
    Strategies: 'Mean Imputation', 'Median Imputation', 'Mode Imputation', 'Forward Fill',
                'Backward Fill', 'Linear Interpolation', 'KNN Imputation', 'Drop Rows', 'Drop Columns (> Threshold)', 'Auto-Select'
    """
    df_clean = df.copy()
    initial_nulls = int(df_clean.isna().sum().sum())
    cols_to_process = custom_cols if custom_cols else df_clean.columns.tolist()

    if strategy == "Drop Columns (> Threshold)":
        drop_cols = [c for c in cols_to_process if (df_clean[c].isna().sum() / len(df_clean)) * 100.0 > threshold_pct]
        df_clean = df_clean.drop(columns=drop_cols)
        final_nulls = int(df_clean.isna().sum().sum())
        return df_clean, {"action": f"Dropped {len(drop_cols)} columns", "initial_nulls": initial_nulls, "final_nulls": final_nulls, "imputed_cells": initial_nulls - final_nulls}

    elif strategy == "Drop Rows":
        df_clean = df_clean.dropna(subset=cols_to_process)
        final_nulls = int(df_clean.isna().sum().sum())
        return df_clean, {"action": "Dropped rows with nulls", "initial_nulls": initial_nulls, "final_nulls": final_nulls, "imputed_cells": initial_nulls - final_nulls}

    elif strategy in ["Mean Imputation", "Median Imputation", "Mode Imputation", "Auto-Select"]:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.intersection(cols_to_process)
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.intersection(cols_to_process)

        for c in num_cols:
            if df_clean[c].isna().sum() > 0:
                val = df_clean[c].median() if (strategy == "Median Imputation" or strategy == "Auto-Select") else df_clean[c].mean()
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

    final_nulls = int(df_clean.isna().sum().sum())
    return df_clean, {
        "strategy": strategy,
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "imputed_cells": max(0, initial_nulls - final_nulls)
    }
