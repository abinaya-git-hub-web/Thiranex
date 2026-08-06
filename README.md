# Thiranex Solutions — Enterprise Predictive Analytics Platform

An interactive, production-ready time-series forecasting and regression suite built with Python, Streamlit, Scikit-learn, Statsmodels, Prophet, XGBoost, and Plotly.

## 📁 Modular Directory Structure

```text
Thiranex_Predictive_Analytics/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── assets/
│   ├── style.css               # Glassmorphic dark theme CSS stylesheet
│   └── logo.svg                # Thiranex Solutions SVG brand logo
└── modules/
    ├── __init__.py             # Module exports
    ├── data_loader.py          # Data import (CSV, Excel, JSON) & profiling
    ├── synthetic_data.py       # 3+ years multi-pattern synthetic generator
    ├── preprocessor.py         # Cleaning, stationarity testing (ADF/KPSS), feature engineering
    ├── model_selector.py       # AutoML suite & weighted top-3 ensemble
    ├── arima_model.py          # ARIMA / SARIMAX time-series implementation
    ├── prophet_model.py        # Facebook Prophet & Fourier fallback
    ├── regression_models.py    # Ridge, Lasso, ElasticNet, Random Forest, XGBoost, SVR
    ├── neural_network.py       # Deep neural network (LSTM sequence surrogate)
    ├── model_evaluator.py      # Metrics (RMSE, MAE, MAPE, R²), residual diagnostics, confidence bands
    ├── chart_builder.py        # 13+ Plotly interactive visualizations
    ├── scenario_analyzer.py    # Interactive what-if business simulation
    ├── insight_generator.py    # Natural language AI executive briefing
    ├── novelty_features.py     # 10 novel predictive features
    ├── filters.py              # Dynamic sidebar filters & scenario controls
    ├── exporters.py            # PDF report, CSV prediction download, REST API payload
    └── ui_components.py        # UI cards, header banner, onboarding tour, glassmorphism theme
```

## 🚀 Key Features

1. **AutoML Engine**: Automatically fits, evaluates, and ranks 8 algorithms, constructing a weighted top-3 ensemble prediction.
2. **13 Interactive Visualizations**: Overlay plots, forecast horizons, residual dashboards, feature importance, Q-Q plots, ACF/PACF, error heatmaps, and walk-forward validation curves.
3. **10 Novel Features**:
   - Model Recommender Engine
   - Prediction Anomaly Monitor (2σ/3σ deviation flags)
   - What-If Scenario Analyzer
   - Risk-Advised Forecasting (P10/P50/P90 interval breakdown)
   - Multi-Variate Prediction Engine
   - AI Executive Briefing Generator
   - Budget Target Variance Analysis
   - External Variable Impact Analyzer
   - Explainable AI Dashboard
   - Adaptive Learning & Model Drift Detector
4. **Enterprise Exports**: Download CSV predictions, ReportLab PDF Executive Briefings, and REST API JSON payloads.

## 🛠️ Quickstart Installation & Execution

```bash
# 1. Clone repository
git clone https://github.com/Thiranex/Predictive-Analytics.git
cd Thiranex

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```
