"""
=============================================================================
Thiranex Solutions — Advanced Data Preprocessing & Feature Engineering
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# 1. Automated Data Cleaning & Smoothing
# ---------------------------------------------------------------------------
def clean_time_series_data(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    impute_method: str = "Linear Interpolation",
    outlier_method: str = "IQR Method",
    smooth_method: str = "None",
    smooth_window: int = 7
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans time-series data: parses dates, sorts chronologically, imputes missing values,
    detects/treats outliers, and applies optional smoothing.
    """
    df_clean = df.copy()

    # Ensure Date column is datetime and sorted
    if date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

    # Impute Missing Values in Target & Numeric Regressors
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    missing_handled = 0
    for col in numeric_cols:
        n_missing = df_clean[col].isnull().sum()
        if n_missing > 0:
            missing_handled += n_missing
            if impute_method == "Linear Interpolation":
                df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()
            elif impute_method == "Forward Fill (ffill)":
                df_clean[col] = df_clean[col].ffill().bfill()
            elif impute_method == "Backward Fill (bfill)":
                df_clean[col] = df_clean[col].bfill().ffill()
            elif impute_method == "Mean Imputation":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif impute_method == "Median Imputation":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Outlier Detection & Handling
    outliers_detected = 0
    if target_col in df_clean.columns and outlier_method != "None":
        s = df_clean[target_col]
        if outlier_method == "IQR Method":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (s < lower_bound) | (s > upper_bound)
            outliers_detected = int(outlier_mask.sum())
            # Cap outliers at boundaries
            df_clean[target_col] = np.clip(s, lower_bound, upper_bound)

        elif outlier_method == "Z-Score Method":
            mean = s.mean()
            std = s.std()
            z_scores = np.abs((s - mean) / (std + 1e-8))
            outlier_mask = z_scores > 3.0
            outliers_detected = int(outlier_mask.sum())
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            df_clean[target_col] = np.clip(s, lower_bound, upper_bound)

        elif outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.03, random_state=42)
            preds = iso.fit_predict(s.values.reshape(-1, 1))
            outlier_mask = preds == -1
            outliers_detected = int(outlier_mask.sum())
            # Replace outliers with rolling median
            rolling_med = s.rolling(window=7, min_periods=1, center=True).median()
            df_clean.loc[outlier_mask, target_col] = rolling_med[outlier_mask]

    # Data Smoothing
    if smooth_method == "Moving Average" and target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].rolling(window=smooth_window, min_periods=1).mean()
    elif smooth_method == "Exponential Smoothing" and target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].ewm(span=smooth_window, adjust=False).mean()

    cleaning_meta = {
        "missing_imputed_count": missing_handled,
        "outliers_detected_count": outliers_detected,
        "impute_method": impute_method,
        "outlier_method": outlier_method,
        "smooth_method": smooth_method
    }

    return df_clean, cleaning_meta


# ---------------------------------------------------------------------------
# 2. Stationarity Testing (ADF & KPSS)
# ---------------------------------------------------------------------------
def perform_stationarity_tests(series: pd.Series) -> Dict[str, Any]:
    """
    Executes Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.
    """
    clean_series = series.dropna()

    adf_result = {"adf_stat": 0.0, "p_value": 1.0, "is_stationary": False}
    try:
        adf_res = adfuller(clean_series, autolag="AIC")
        adf_result = {
            "adf_stat": float(adf_res[0]),
            "p_value": float(adf_res[1]),
            "is_stationary": float(adf_res[1]) < 0.05,
            "critical_values": {k: float(v) for k, v in adf_res[4].items()}
        }
    except Exception:
        pass

    kpss_result = {"kpss_stat": 0.0, "p_value": 0.0, "is_stationary": False}
    try:
        kpss_res = kpss(clean_series, regression="c", nlags="auto")
        kpss_result = {
            "kpss_stat": float(kpss_res[0]),
            "p_value": float(kpss_res[1]),
            "is_stationary": float(kpss_res[1]) >= 0.05,
            "critical_values": {k: float(v) for k, v in kpss_res[3].items()}
        }
    except Exception:
        pass

    # Overall Stationarity Verdict
    if adf_result["is_stationary"] and kpss_result["is_stationary"]:
        verdict = "Stationary (Series has no unit root and constant variance)"
    elif not adf_result["is_stationary"] and not kpss_result["is_stationary"]:
        verdict = "Non-Stationary (Trend or strong seasonal component present)"
    elif adf_result["is_stationary"] and not kpss_result["is_stationary"]:
        verdict = "Difference Stationary (Requires 1st differencing)"
    else:
        verdict = "Trend Stationary (Requires linear trend removal)"

    return {
        "adf": adf_result,
        "kpss": kpss_result,
        "verdict": verdict
    }


# ---------------------------------------------------------------------------
# 3. Feature Engineering Engine
# ---------------------------------------------------------------------------
def engineer_time_series_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    numeric_features: List[str] = None,
    max_lags: int = 7,
    rolling_windows: List[int] = [7, 14, 30],
    include_calendar: bool = True,
    include_fourier: bool = True,
    include_diffs: bool = True
) -> pd.DataFrame:
    """
    Constructs ML regressors from time series:
    - Lag features (1..max_lags)
    - Rolling window statistics (mean, std, min, max, median)
    - Date/Time components (Year, Quarter, Month, Week, Day, DayOfWeek, IsWeekend)
    - Seasonal Fourier sine/cosine terms
    - Difference features (1st & 2nd differences)
    - Interaction terms between numeric regressors
    """
    df_feat = df.copy()

    # A. Date/Time Features
    if date_col in df_feat.columns and include_calendar:
        dt_s = pd.to_datetime(df_feat[date_col])
        df_feat["Year"] = dt_s.dt.year
        df_feat["Quarter"] = dt_s.dt.quarter
        df_feat["Month"] = dt_s.dt.month
        df_feat["WeekOfYear"] = dt_s.dt.isocalendar().week.astype(int)
        df_feat["DayOfMonth"] = dt_s.dt.day
        df_feat["DayOfWeek"] = dt_s.dt.dayofweek
        df_feat["Is_Weekend"] = dt_s.dt.dayofweek.isin([5, 6]).astype(int)

        # Fourier Terms for Annual & Weekly Seasonality
        if include_fourier:
            day_of_year = dt_s.dt.dayofyear
            df_feat["Fourier_Sin_Year"] = np.sin(2 * np.pi * day_of_year / 365.25)
            df_feat["Fourier_Cos_Year"] = np.cos(2 * np.pi * day_of_year / 365.25)
            df_feat["Fourier_Sin_Week"] = np.sin(2 * np.pi * dt_s.dt.dayofweek / 7.0)
            df_feat["Fourier_Cos_Week"] = np.cos(2 * np.pi * dt_s.dt.dayofweek / 7.0)

    # B. Target Lags & Differences
    if target_col in df_feat.columns:
        s = df_feat[target_col]
        for lag in range(1, max_lags + 1):
            df_feat[f"Lag_{lag}"] = s.shift(lag)

        if include_diffs:
            df_feat["Diff_1"] = s.diff(1)
            df_feat["Diff_2"] = s.diff(2)

        # Rolling Statistics over window sizes
        for w in rolling_windows:
            if len(df_feat) >= w:
                df_feat[f"Rolling_Mean_{w}"] = s.shift(1).rolling(window=w).mean()
                df_feat[f"Rolling_Std_{w}"] = s.shift(1).rolling(window=w).std()
                df_feat[f"Rolling_Min_{w}"] = s.shift(1).rolling(window=w).min()
                df_feat[f"Rolling_Max_{w}"] = s.shift(1).rolling(window=w).max()

    # C. Interaction Features (if multiple numeric features exist)
    num_cols = numeric_features if numeric_features else []
    if len(num_cols) >= 2:
        col1, col2 = num_cols[0], num_cols[1]
        if col1 in df_feat.columns and col2 in df_feat.columns:
            df_feat[f"Interaction_{col1}_x_{col2}"] = df_feat[col1] * df_feat[col2]

    # Drop rows with NaNs introduced by lagging
    df_feat = df_feat.bfill().ffill()

    return df_feat


# ---------------------------------------------------------------------------
# 4. Train-Test Time-Based Split Generator
# ---------------------------------------------------------------------------
def split_time_series(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological time-based split.
    """
    n = len(df)
    split_idx = int(n * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df
