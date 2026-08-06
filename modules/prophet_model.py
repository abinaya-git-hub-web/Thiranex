"""
=============================================================================
Thiranex Solutions — Facebook Prophet Model Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def fit_predict_prophet(
    df_train: pd.DataFrame,
    date_col: str,
    target_col: str,
    test_horizon: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits Facebook Prophet model (or Fourier Trend fallback if Prophet is not installed).
    """
    if HAS_PROPHET:
        try:
            m_df = pd.DataFrame({
                "ds": pd.to_datetime(df_train[date_col]),
                "y": df_train[target_col].values
            })
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(m_df)

            future = model.make_future_dataframe(periods=test_horizon)
            forecast_df = model.predict(future)

            fitted = forecast_df["yhat"].values[:len(df_train)]
            forecast = forecast_df["yhat"].values[len(df_train):]
            lower = forecast_df["yhat_lower"].values[len(df_train):]
            upper = forecast_df["yhat_upper"].values[len(df_train):]

            meta = {"conf_lower": lower, "conf_upper": upper, "engine": "Facebook Prophet"}
            return forecast, fitted, meta
        except Exception:
            pass

    # Prophet Fallback: Fourier Trend & Seasonality Curve Fit
    y_train = df_train[target_col].values
    n = len(y_train)
    t = np.arange(n)
    t_fut = np.arange(n, n + test_horizon)

    X_tr = np.column_stack([t, np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25), np.sin(2 * np.pi * t / 7.0)])
    X_fut = np.column_stack([t_fut, np.sin(2 * np.pi * t_fut / 365.25), np.cos(2 * np.pi * t_fut / 365.25), np.sin(2 * np.pi * t_fut / 7.0)])

    reg = Ridge(alpha=1.0).fit(X_tr, y_train)
    fitted = reg.predict(X_tr)
    forecast = reg.predict(X_fut)

    std_err = np.std(y_train - fitted)
    meta = {
        "conf_lower": forecast - 1.96 * std_err,
        "conf_upper": forecast + 1.96 * std_err,
        "engine": "Prophet Fallback (Fourier Decomposition)"
    }
    return forecast, fitted, meta
