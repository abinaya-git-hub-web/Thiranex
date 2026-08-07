"""
=============================================================================
Thiranex Solutions — Outlier Detection & Handling Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest

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
