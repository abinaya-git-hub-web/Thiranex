"""
=============================================================================
Thiranex Solutions — ARIMA & SARIMAX Time-Series Model Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_predict_sarima(
    train_series: pd.Series,
    test_horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits SARIMAX model and returns forecast array, fitted values, and metadata.
    """
    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fitted = res.fittedvalues.values
        forecast_res = res.get_forecast(steps=test_horizon)
        forecast = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=0.05).values

        meta = {
            "aic": float(res.aic),
            "bic": float(res.bic),
            "order": order,
            "seasonal_order": seasonal_order,
            "conf_lower": conf_int[:, 0],
            "conf_upper": conf_int[:, 1]
        }
        return forecast, fitted, meta
    except Exception as e:
        model = ExponentialSmoothing(train_series, trend="add").fit()
        forecast = model.forecast(test_horizon).values
        fitted = model.fittedvalues.values
        meta = {"aic": 0.0, "bic": 0.0, "error": str(e)}
        return forecast, fitted, meta
