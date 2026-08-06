"""
=============================================================================
Thiranex Solutions — What-If Scenario Analyzer Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple


def simulate_what_if_scenario(
    base_forecast: np.ndarray,
    marketing_change_pct: float = 0.0,
    price_change_pct: float = 0.0,
    competitor_impact_pct: float = 0.0,
    marketing_elasticity: float = 0.35,
    price_elasticity: float = -1.2,
    competitor_elasticity: float = -0.4
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulates business decision modifications (e.g. +15% marketing spend, -5% pricing shift).
    """
    marketing_factor = 1.0 + (marketing_change_pct / 100.0) * marketing_elasticity
    price_factor = 1.0 + (price_change_pct / 100.0) * price_elasticity
    competitor_factor = 1.0 + (competitor_impact_pct / 100.0) * competitor_elasticity

    combined_multiplier = marketing_factor * price_factor * competitor_factor
    simulated_forecast = base_forecast * combined_multiplier

    base_total = float(np.sum(base_forecast))
    simulated_total = float(np.sum(simulated_forecast))
    net_delta = simulated_total - base_total
    pct_change = round((net_delta / max(base_total, 1e-8)) * 100, 2)

    summary = {
        "base_total": round(base_total, 2),
        "simulated_total": round(simulated_total, 2),
        "net_delta": round(net_delta, 2),
        "pct_change": pct_change
    }

    return simulated_forecast, summary
