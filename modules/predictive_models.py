"""
=============================================================================
Thiranex Solutions — Multi-Model Forecasting & AutoML Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple, List, Optional

# Core Statsmodels & Sklearn imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Graceful optional imports for XGBoost & Prophet
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Time-Series Forecasting Models
# ---------------------------------------------------------------------------
def fit_predict_sarima(
    train_series: pd.Series,
    test_horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits SARIMAX model and returns forecast array, fitted train array, and metrics.
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
        # Fallback simple exponential smoothing fit
        model = ExponentialSmoothing(train_series, trend="add").fit()
        forecast = model.forecast(test_horizon).values
        fitted = model.fittedvalues.values
        meta = {"aic": 0.0, "bic": 0.0, "error": str(e)}
        return forecast, fitted, meta


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


def fit_predict_holt_winters(
    train_series: pd.Series,
    test_horizon: int,
    seasonal_periods: int = 7
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits Holt-Winters Exponential Smoothing.
    """
    try:
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add" if len(train_series) >= 2 * seasonal_periods else None,
            seasonal_periods=seasonal_periods if len(train_series) >= 2 * seasonal_periods else None
        ).fit()

        fitted = model.fittedvalues.values
        forecast = model.forecast(test_horizon).values
        std_err = np.std(train_series.values - fitted)

        meta = {
            "conf_lower": forecast - 1.96 * std_err,
            "conf_upper": forecast + 1.96 * std_err,
            "aic": float(model.aic)
        }
        return forecast, fitted, meta
    except Exception as e:
        # Fallback simple trend
        model = ExponentialSmoothing(train_series, trend="add").fit()
        fitted = model.fittedvalues.values
        forecast = model.forecast(test_horizon).values
        meta = {"conf_lower": forecast, "conf_upper": forecast, "error": str(e)}
        return forecast, fitted, meta


def fit_predict_var(
    df_train: pd.DataFrame,
    target_cols: List[str],
    test_horizon: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Fits Vector Autoregression (VAR) for multi-series predictions.
    """
    data = df_train[target_cols].dropna().values
    try:
        model = VAR(data)
        res = model.fit(maxlags=5)
        lag_order = res.k_ar
        forecast_matrix = res.forecast(data[-lag_order:], steps=test_horizon)
        fitted_matrix = res.fittedvalues

        forecasts = {col: forecast_matrix[:, i] for i, col in enumerate(target_cols)}
        fitted = {col: fitted_matrix[:, i] for i, col in enumerate(target_cols)}
        meta = {"aic": float(res.aic), "bic": float(res.bic), "lag_order": lag_order}
        return forecasts, fitted, meta
    except Exception as e:
        # Fallback: independent univariate models
        forecasts, fitted = {}, {}
        for col in target_cols:
            f, fit, _ = fit_predict_holt_winters(df_train[col], test_horizon)
            forecasts[col] = f
            fitted[col] = fit
        return forecasts, fitted, {"error": str(e)}


# ---------------------------------------------------------------------------
# 2. Supervised ML Regression Models
# ---------------------------------------------------------------------------
def fit_predict_regression_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Fits supervised ML regression models: Ridge, Lasso, ElasticNet, Random Forest,
    XGBoost, SVR, or Neural Network (MLPRegressor).
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train.fillna(0))
    X_te_scaled = scaler.transform(X_test.fillna(0))

    if model_type == "Ridge Regression":
        model = Ridge(alpha=1.0)
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=0.1)
    elif model_type == "ElasticNet":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    elif model_type == "XGBoost Regressor":
        if HAS_XGB:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    elif model_type == "Support Vector Regressor (SVR)":
        model = SVR(C=10.0, epsilon=0.1, kernel="rbf")
    elif model_type == "Neural Network (LSTM / MLP)":
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_tr_scaled, y_train)

    fitted = model.predict(X_tr_scaled)
    forecast = model.predict(X_te_scaled)

    return forecast, fitted, model


# ---------------------------------------------------------------------------
# 3. AutoML & Ensemble Selection Engine
# ---------------------------------------------------------------------------
def run_automl_suite(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Automatically trains all available time-series and ML regression models,
    evaluates RMSE/MAE/MAPE on test set, ranks models, picks best recommender model,
    and calculates a weighted top-3 ensemble prediction.
    """
    test_horizon = len(df_test)
    y_test = df_test[target_col].values

    models_to_test = [
        "SARIMA / SARIMAX",
        "Prophet (Facebook)",
        "Holt-Winters Exponential Smoothing",
        "Ridge Regression",
        "Random Forest",
        "XGBoost Regressor",
        "Support Vector Regressor (SVR)",
        "Neural Network (LSTM / MLP)"
    ]

    results = {}
    forecasts_dict = {}

    # Prepare ML Regression datasets
    X_train_df = df_train[feature_cols] if feature_cols else pd.DataFrame({"t": np.arange(len(df_train))})
    X_test_df = df_test[feature_cols] if feature_cols else pd.DataFrame({"t": np.arange(len(df_train), len(df_train) + test_horizon)})
    y_train_s = df_train[target_col]

    for model_name in models_to_test:
        try:
            if model_name == "SARIMA / SARIMAX":
                fc, fit, meta = fit_predict_sarima(y_train_s, test_horizon)
            elif model_name == "Prophet (Facebook)":
                fc, fit, meta = fit_predict_prophet(df_train, date_col, target_col, test_horizon)
            elif model_name == "Holt-Winters Exponential Smoothing":
                fc, fit, meta = fit_predict_holt_winters(y_train_s, test_horizon)
            else:
                fc, fit, model_obj = fit_predict_regression_model(model_name, X_train_df, y_train_s, X_test_df)

            # Compute Evaluation Metrics
            rmse = float(np.sqrt(np.mean((y_test - fc) ** 2)))
            mae = float(np.mean(np.abs(y_test - fc)))
            mape = float(np.mean(np.abs((y_test - fc) / np.maximum(np.abs(y_test), 1e-8))) * 100)

            results[model_name] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "forecast": fc,
                "fitted": fit
            }
            forecasts_dict[model_name] = fc
        except Exception as e:
            pass

    # Sort models by RMSE ascending
    ranked_models = sorted(results.items(), key=lambda x: x[1]["rmse"])
    best_model_name = ranked_models[0][0] if ranked_models else "Ridge Regression"

    # Compute Ensemble Forecast (Weighted average of top 3 models)
    top_3 = ranked_models[:min(3, len(ranked_models))]
    if top_3:
        inv_rmses = [1.0 / (m[1]["rmse"] + 1e-8) for m in top_3]
        weights = [w / sum(inv_rmses) for w in inv_rmses]

        ensemble_fc = np.zeros(test_horizon)
        for idx, (m_name, m_data) in enumerate(top_3):
            ensemble_fc += weights[idx] * m_data["forecast"]

        ens_rmse = float(np.sqrt(np.mean((y_test - ensemble_fc) ** 2)))
        ens_mae = float(np.mean(np.abs(y_test - ensemble_fc)))
        ens_mape = float(np.mean(np.abs((y_test - ensemble_fc) / np.maximum(np.abs(y_test), 1e-8))) * 100)

        results["Ensemble (Top 3 Weighted)"] = {
            "rmse": ens_rmse,
            "mae": ens_mae,
            "mape": ens_mape,
            "forecast": ensemble_fc,
            "fitted": top_3[0][1]["fitted"]
        }

    return {
        "leaderboard": results,
        "best_model_name": best_model_name,
        "ranked_models": ranked_models
    }
