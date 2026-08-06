"""
=============================================================================
Thiranex Solutions — Synthetic Time-Series Data Generator
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple


def generate_synthetic_timeseries(
    scenario: str = "Sales Data",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    trend_slope: float = 0.05,
    noise_level: float = 15.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates 3+ years of realistic time-series data with multi-pattern dynamics:
    - Base trend (linear growth)
    - Seasonality (yearly, quarterly, monthly, weekly)
    - Noise (random Gaussian variation)
    - Holiday spikes & dips
    - Synthetic regressors (Marketing Spend, Competitor Activity, Region, Product Category)
    """
    np.random.seed(random_seed)

    # Date range generation
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    t = np.arange(n)

    # 1. Base Linear Trend
    base_level = 500.0
    trend = base_level + (trend_slope * t)

    # 2. Seasonality Components
    yearly_seasonality = 120.0 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    quarterly_seasonality = 45.0 * np.cos(2 * np.pi * t / 91.25)
    monthly_seasonality = 30.0 * np.sin(2 * np.pi * t / 30.44)
    weekly_seasonality = 25.0 * np.sin(2 * np.pi * t / 7.0)

    # 3. Holiday Effects & Special Events (Spikes around Nov-Dec, Dips in Jan)
    holiday_effect = np.zeros(n)
    for i, date in enumerate(dates):
        # Black Friday / Cyber Monday / Holiday season spike (late Nov to Dec)
        if date.month == 11 and date.day >= 20:
            holiday_effect[i] += np.random.uniform(150, 300)
        elif date.month == 12 and date.day <= 25:
            holiday_effect[i] += np.random.uniform(100, 250)
        elif date.month == 1 and date.day <= 5:
            holiday_effect[i] -= np.random.uniform(50, 100) # Post-holiday dip
        elif date.month == 7 and date.day in [4, 5]: # Summer promo spike
            holiday_effect[i] += np.random.uniform(80, 160)

    # 4. Noise
    noise = np.random.normal(0, noise_level, size=n)

    # 5. External Predictors (Synthetic Regressors)
    # Marketing spend correlated with target + noise
    marketing_spend = np.round(np.clip(1000 + 0.3 * t + 80 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 150, n), 200, 5000), 2)
    competitor_activity = np.round(np.clip(50 + 20 * np.cos(2 * np.pi * t / 180) + np.random.normal(0, 10, n), 10, 100), 1)

    # Build target variable based on scenario selection
    if scenario == "Sales Data":
        # Target: Daily Sales (Units)
        target_series = trend + yearly_seasonality + monthly_seasonality + (marketing_spend * 0.08) - (competitor_activity * 1.2) + holiday_effect + noise
        target_col = "Sales_Units"
    elif scenario == "Revenue Data":
        # Target: Daily Revenue ($)
        target_series = (trend * 3.5) + (yearly_seasonality * 4) + (quarterly_seasonality * 3) + (marketing_spend * 0.25) - (competitor_activity * 4.0) + (holiday_effect * 5) + (noise * 3)
        target_col = "Revenue_USD"
    else: # Website Traffic
        # Target: Daily Page Views / Visitors
        target_series = (trend * 10) + (weekly_seasonality * 80) + (yearly_seasonality * 15) + (marketing_spend * 0.6) + (holiday_effect * 12) + (noise * 8)
        target_col = "Website_Traffic"

    # Ensure strictly positive target values
    target_series = np.clip(target_series, a_min=10.0, a_max=None)

    # Auxiliary secondary targets for multi-variate forecasting
    secondary_revenue = np.round(target_series * np.random.uniform(25.0, 35.0, size=n) + np.random.normal(0, 100, size=n), 2)
    secondary_units = np.round(target_series * np.random.uniform(0.8, 1.2, size=n), 0)
    secondary_profit = np.round(secondary_revenue * 0.32 + np.random.normal(0, 50, size=n), 2)

    # Categories and Regions
    categories = ["Electronics", "Enterprise Software", "Cloud Services", "Hardware"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]

    df = pd.DataFrame({
        "Date": dates,
        target_col: np.round(target_series, 2),
        "Revenue_USD": secondary_revenue,
        "Units_Sold": secondary_units,
        "Profit_USD": secondary_profit,
        "Marketing_Spend": marketing_spend,
        "Competitor_Activity_Index": competitor_activity,
        "Product_Category": np.random.choice(categories, size=n),
        "Region": np.random.choice(regions, size=n)
    })

    return df
