"""
=============================================================================
Thiranex Solutions — AI Insight & Executive Summary Generator Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_ai_executive_summary(
    target_name: str,
    best_model_name: str,
    metrics: Dict[str, float],
    forecast_sum: float,
    historic_sum: float,
    risk_info: Dict[str, Any],
    anomalies_count: int
) -> str:
    """
    Generates natural language automated executive briefing of forecast, performance, and risk.
    """
    growth_pct = round(((forecast_sum - historic_sum) / max(historic_sum, 1e-8)) * 100, 1)
    direction = "growth 📈" if growth_pct >= 0 else "contraction 📉"

    summary_md = f"""
    ### 🎯 Thiranex AI Executive Briefing — {target_name}

    **1. Forecast Direction & Projections:**
    Over the next forecast horizon, predicted **{target_name}** totals **${forecast_sum:,.2f}** (or units), representing a **{abs(growth_pct)}% {direction}** compared to the prior baseline period (${historic_sum:,.2f}).

    **2. Optimal Model Selection:**
    The automated AutoML engine selected **{best_model_name}** as the top-performing algorithm. It achieved an outstanding **MAPE of {metrics.get('mape', 0.0)}%** and a Root Mean Squared Error (RMSE) of **{metrics.get('rmse', 0.0):,.2f}**, ensuring high analytical reliability.

    **3. Risk & Anomaly Assessment:**
    - **Risk Status**: {risk_info.get('risk_level', 'Low Risk')}
    - **Uncertainty Spread**: {risk_info.get('relative_uncertainty_pct', 0.0)}%
    - **Historical Anomalies**: Identified **{anomalies_count}** anomalous data points exceeding standard tolerance threshold.

    **4. Strategic Recommendation:**
    {risk_info.get('advice', 'Proceed with baseline strategic plan while monitoring key operational KPIs.')}
    """
    return summary_md.strip()
