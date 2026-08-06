"""
=============================================================================
Thiranex Solutions — AutoML Model Selector & Ensemble Engine Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from modules.arima_model import fit_predict_sarima
from modules.prophet_model import fit_predict_prophet
from modules.regression_models import fit_predict_regression_model
from modules.neural_network import fit_predict_neural_network
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_predict_holt_winters(
    train_series: pd.Series,
    test_horizon: int,
    seasonal_periods: int = 7
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fits Holt-Winters Exponential Smoothing."""
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
        model = ExponentialSmoothing(train_series, trend="add").fit()
        fitted = model.fittedvalues.values
        forecast = model.forecast(test_horizon).values
        meta = {"conf_lower": forecast, "conf_upper": forecast, "error": str(e)}
        return forecast, fitted, meta


def run_automl_suite(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Automatically trains all available models, computes out-of-sample RMSE/MAE/MAPE,
    ranks models, picks optimal recommender, and builds a weighted top-3 ensemble.
    """
    test_horizon = len(df_test)
    y_test = df_test[target_col].values

    models_to_test = [
        "SARIMA / SARIMAX",
        "Prophet (Facebook)",
        "Holt-Winters Exponential Smoothing",
        "Ridge Regression",
        "Random Forest Regressor",
        "XGBoost Regressor",
        "Support Vector Regressor (SVR)",
        "Neural Network (LSTM / MLP)"
    ]

    results = {}
    forecasts_dict = {}

    num_feature_cols = [c for c in feature_cols if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c])]

    X_train_df = df_train[num_feature_cols] if num_feature_cols else pd.DataFrame({"t": np.arange(len(df_train))})
    X_test_df = df_test[num_feature_cols] if num_feature_cols else pd.DataFrame({"t": np.arange(len(df_train), len(df_train) + test_horizon)})
    y_train_s = df_train[target_col]

    for model_name in models_to_test:
        try:
            if model_name == "SARIMA / SARIMAX":
                fc, fit, meta = fit_predict_sarima(y_train_s, test_horizon)
            elif model_name == "Prophet (Facebook)":
                fc, fit, meta = fit_predict_prophet(df_train, date_col, target_col, test_horizon)
            elif model_name == "Holt-Winters Exponential Smoothing":
                fc, fit, meta = fit_predict_holt_winters(y_train_s, test_horizon)
            elif model_name == "Neural Network (LSTM / MLP)":
                fc, fit, model_obj = fit_predict_neural_network(X_train_df, y_train_s, X_test_df)
            else:
                fc, fit, model_obj = fit_predict_regression_model(model_name, X_train_df, y_train_s, X_test_df)

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
        except Exception:
            pass

    sorted_models = sorted(results.items(), key=lambda x: x[1]["rmse"])
    best_model_name = sorted_models[0][0] if sorted_models else "Ridge Regression"

    # Compute Ensemble Forecast (Weighted average of top 3 models)
    top_3 = sorted_models[:min(3, len(sorted_models))]
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
        "ranked_models": sorted_models
    }
