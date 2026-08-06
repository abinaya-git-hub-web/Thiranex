"""
=============================================================================
Thiranex Solutions — Supervised ML Regression Models Module
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def fit_predict_regression_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Fits supervised regression models: Ridge, Lasso, ElasticNet, Random Forest,
    XGBoost, or SVR.
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
    elif model_type == "Random Forest" or model_type == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    elif model_type == "XGBoost Regressor":
        if HAS_XGB:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    elif model_type == "Support Vector Regressor (SVR)":
        model = SVR(C=10.0, epsilon=0.1, kernel="rbf")
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_tr_scaled, y_train)

    fitted = model.predict(X_tr_scaled)
    forecast = model.predict(X_te_scaled)

    return forecast, fitted, model
