"""
=============================================================================
Thiranex Solutions — Model Evaluation & Residual Diagnostics Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ---------------------------------------------------------------------------
# 1. Performance Metrics Engine
# ---------------------------------------------------------------------------
def calculate_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray = None,
    num_params: int = 1
) -> Dict[str, float]:
    """
    Computes complete set of performance metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - SMAPE (Symmetric MAPE)
    - MASE (Mean Absolute Scaled Error)
    - R² Score
    - Adjusted R²
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    # 1. RMSE & MAE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # 2. MAPE
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

    # 3. SMAPE
    smape_denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.abs(y_true - y_pred) / np.maximum(smape_denom, 1e-8)) * 100.0

    # 4. MASE (Mean Absolute Scaled Error)
    mase = 1.0
    if y_train is not None and len(y_train) > 1:
        naive_mae = np.mean(np.abs(np.diff(y_train)))
        if naive_mae > 1e-8:
            mase = mae / naive_mae

    # 5. R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / np.maximum(ss_tot, 1e-8))

    # 6. Adjusted R²
    adj_r2 = r2
    if n > num_params + 1:
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - num_params - 1))

    return {
        "rmse": float(np.round(rmse, 4)),
        "mae": float(np.round(mae, 4)),
        "mape": float(np.round(mape, 2)),
        "smape": float(np.round(smape, 2)),
        "mase": float(np.round(mase, 3)),
        "r2": float(np.round(r2, 4)),
        "adj_r2": float(np.round(adj_r2, 4))
    }


# ---------------------------------------------------------------------------
# 2. Residual Analysis Diagnostics
# ---------------------------------------------------------------------------
def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Analyzes residuals for normality, autocorrelation, and independence:
    - Distribution stats (mean, std, skewness, kurtosis)
    - Q-Q plot quantiles
    - ACF & PACF values
    - Ljung-Box test for independence
    """
    residuals = y_true - y_pred
    n = len(residuals)

    # 1. Summary Stats
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    skew_res = float(stats.skew(residuals)) if n > 3 else 0.0
    kurt_res = float(stats.kurtosis(residuals)) if n > 3 else 0.0

    # 2. Q-Q Plot theoretical vs sample quantiles
    qq_sample = np.sort(residuals)
    qq_theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    # 3. ACF & PACF
    nlags = min(20, n // 2 - 1) if n > 10 else 5
    acf_vals = acf(residuals, nlags=nlags, fft=True)
    pacf_vals = pacf(residuals, nlags=nlags, method="ywm")

    # 4. Ljung-Box Test
    lb_pvalue = 1.0
    lb_stat = 0.0
    try:
        lb_res = acorr_ljungbox(residuals, lags=[min(10, nlags)], return_df=True)
        lb_stat = float(lb_res["lb_stat"].values[0])
        lb_pvalue = float(lb_res["lb_pvalue"].values[0])
    except Exception:
        pass

    return {
        "residuals": residuals,
        "mean": mean_res,
        "std": std_res,
        "skewness": skew_res,
        "kurtosis": kurt_res,
        "qq_sample": qq_sample,
        "qq_theoretical": qq_theoretical,
        "acf": acf_vals,
        "pacf": pacf_vals,
        "ljung_box": {
            "statistic": lb_stat,
            "p_value": lb_pvalue,
            "is_independent": lb_pvalue >= 0.05
        }
    }


# ---------------------------------------------------------------------------
# 3. Prediction Intervals & Confidence Bands
# ---------------------------------------------------------------------------
def compute_prediction_intervals(
    y_pred: np.ndarray,
    residual_std: float,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes lower and upper prediction intervals based on standard error.
    Confidence levels: 0.80 (z=1.282), 0.90 (z=1.645), 0.95 (z=1.960), 0.99 (z=2.576)
    """
    z_map = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence_level, 1.960)

    # Uncertainty grows slightly into the future horizon
    h = len(y_pred)
    expansion_factor = np.sqrt(1.0 + (np.arange(h) / max(h, 1)) * 0.5)

    margin = z * residual_std * expansion_factor
    lower_bound = np.maximum(y_pred - margin, 0.0) # Ensure non-negative if business domain
    upper_bound = y_pred + margin

    return lower_bound, upper_bound
