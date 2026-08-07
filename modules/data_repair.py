"""
=============================================================================
Thiranex Solutions — AI Data Repair & ML Imputation Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

def ml_predictive_imputation(df: pd.DataFrame, target_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Uses Random Forest ML models to predict and fill missing values in numeric fields.
    """
    df_clean = df.copy()
    num_cols = target_cols if target_cols else df_clean.select_dtypes(include=[np.number]).columns.tolist()
    imputed_count = 0

    for c in num_cols:
        null_cnt = int(df_clean[c].isna().sum())
        if null_cnt > 0:
            other_num = [col for col in num_cols if col != c and df_clean[col].isna().sum() == 0]
            if other_num:
                train_data = df_clean[df_clean[c].notna()]
                test_data = df_clean[df_clean[c].isna()]
                rf = RandomForestRegressor(n_estimators=30, random_state=42)
                rf.fit(train_data[other_num], train_data[c])
                preds = rf.predict(test_data[other_num])
                df_clean.loc[df_clean[c].isna(), c] = preds
                imputed_count += null_cnt
            else:
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
                imputed_count += null_cnt

    return df_clean, {"ml_imputed_cells": imputed_count, "algorithm": "RandomForestRegressor"}
