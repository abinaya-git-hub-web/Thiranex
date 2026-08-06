"""
=============================================================================
Thiranex Solutions — Deep Neural Network & LSTM Sequence Predictor
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def fit_predict_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Fits multi-layer sequence neural network (MLP/LSTM sequence surrogate).
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train.fillna(0))
    X_te_scaled = scaler.transform(X_test.fillna(0))

    # Deep Neural Architecture with 2 hidden layers (64, 32)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=350,
        random_state=42,
        early_stopping=True
    )

    model.fit(X_tr_scaled, y_train)

    fitted = model.predict(X_tr_scaled)
    forecast = model.predict(X_te_scaled)

    return forecast, fitted, model
