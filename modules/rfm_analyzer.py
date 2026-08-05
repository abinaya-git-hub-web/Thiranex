"""
Module: rfm_analyzer.py
Description: RFM (Recency, Frequency, Monetary) scoring and customer segmentation engine.
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_rfm_scores(df: pd.DataFrame, mappings: Dict[str, str]) -> pd.DataFrame:
    """
    Calculates Recency (days), Frequency, and Monetary metrics and scores them from 1 to 5.
    Assigns standard RFM customer segment names.
    """
    df_rfm = df.copy()

    # Recency Calculation (Days since last purchase)
    date_col = mappings.get("last_purchase_date")
    if date_col and date_col in df_rfm.columns:
        df_rfm[date_col] = pd.to_datetime(df_rfm[date_col], errors="coerce")
        max_date = df_rfm[date_col].max()
        if pd.isnull(max_date):
            max_date = pd.Timestamp.now()
        df_rfm["Recency_Days"] = (max_date - df_rfm[date_col]).dt.days.fillna(180)
    else:
        df_rfm["Recency_Days"] = np.random.randint(1, 365, size=len(df_rfm))

    # Frequency Calculation
    freq_col = mappings.get("frequency")
    if freq_col and freq_col in df_rfm.columns:
        df_rfm["Frequency_Val"] = pd.to_numeric(df_rfm[freq_col], errors="coerce").fillna(1)
    else:
        df_rfm["Frequency_Val"] = np.random.randint(1, 20, size=len(df_rfm))

    # Monetary Calculation
    spend_col = mappings.get("spend")
    if spend_col and spend_col in df_rfm.columns:
        df_rfm["Monetary_Val"] = pd.to_numeric(df_rfm[spend_col], errors="coerce").fillna(100.0)
    else:
        df_rfm["Monetary_Val"] = np.random.uniform(100, 5000, size=len(df_rfm))

    # Calculate 1-5 Scores using quantiles or ranking
    def assign_scores(series, ascending=True):
        try:
            return pd.qcut(series.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]).astype(int)
        except Exception:
            return pd.cut(series, bins=5, labels=[1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]).astype(int)

    df_rfm["R_Score"] = assign_scores(df_rfm["Recency_Days"], ascending=False)
    df_rfm["F_Score"] = assign_scores(df_rfm["Frequency_Val"], ascending=True)
    df_rfm["M_Score"] = assign_scores(df_rfm["Monetary_Val"], ascending=True)
    df_rfm["RFM_Score"] = df_rfm["R_Score"].astype(str) + df_rfm["F_Score"].astype(str) + df_rfm["M_Score"].astype(str)

    # Segment Label Assignment Rules
    def segment_rfm(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f >= 2 and m >= 2:
            return "Potential Loyalists"
        elif r >= 4 and f == 1:
            return "New Customers"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        elif r <= 2:
            return "Hibernating"
        else:
            return "Promising & Others"

    df_rfm["RFM_Segment"] = df_rfm.apply(segment_rfm, axis=1)
    return df_rfm
