"""
Module: kpi_calculator.py
Description: Calculates core customer dataset KPIs and segmentation summary statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_kpis(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Computes dataset high-level KPIs for customer segmentation.
    """
    if df.empty:
        return {
            "total_customers": 0,
            "total_spend": 0.0,
            "avg_spend": 0.0,
            "avg_aov": 0.0,
            "avg_frequency": 0.0,
            "avg_satisfaction": 0.0
        }

    total_customers = len(df)

    spend_c = mappings.get("spend") if mappings and mappings.get("spend") in df.columns else ("Total Spend ($)" if "Total Spend ($)" in df.columns else "Monetary_Val")
    total_spend = float(df[spend_c].sum()) if spend_c in df.columns else 0.0
    avg_spend = total_spend / total_customers if total_customers > 0 else 0.0

    aov_c = mappings.get("aov") if mappings and mappings.get("aov") in df.columns else ("Average Order Value ($)" if "Average Order Value ($)" in df.columns else "AOV")
    avg_aov = float(df[aov_c].mean()) if aov_c in df.columns else avg_spend

    freq_c = mappings.get("frequency") if mappings and mappings.get("frequency") in df.columns else ("Purchase Frequency" if "Purchase Frequency" in df.columns else "Frequency_Val")
    avg_frequency = float(df[freq_c].mean()) if freq_c in df.columns else 1.0

    sat_c = mappings.get("satisfaction") if mappings and mappings.get("satisfaction") in df.columns else ("Satisfaction Score" if "Satisfaction Score" in df.columns else None)
    avg_satisfaction = float(df[sat_c].mean()) if sat_c and sat_c in df.columns else 4.0

    return {
        "total_customers": total_customers,
        "total_spend": total_spend,
        "avg_spend": round(avg_spend, 2),
        "avg_aov": round(avg_aov, 2),
        "avg_frequency": round(avg_frequency, 1),
        "avg_satisfaction": round(avg_satisfaction, 1)
    }
