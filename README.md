# 📊 Thiranex Solutions — Enterprise Sales & Revenue Analysis Dashboard

A state-of-the-art, production-ready enterprise analytics dashboard built with **Streamlit**, **Plotly**, **Pandas**, **Scikit-Learn**, **Statsmodels**, and **ReportLab**.

Designed for executive decision-makers, sales leaders, and financial analysts at **Thiranex Solutions**, this application delivers real-time sales performance tracking, AI-powered anomaly detection, predictive revenue forecasting, and PDF report exports.

---

## 🌟 Key Features

### 1. Data Import & Smart Auto-Detection
- **Multi-Format Ingestion**: Supports CSV, Excel (`.xlsx`, `.xls`), and JSON files.
- **Smart Column Detection**: Automatic semantic role recognition for Dates, Revenue, Quantities, Unit Prices, Products, Categories, Regions, Salespersons, Customers, and Payment Methods.
- **Data Profiling**: Instant dataset diagnostics (missing values, data types, unique counts, memory footprint).
- **Synthetic Data Engine**: Generates a 500-row realistic sales dataset when no file is uploaded.

### 2. Advanced KPI Cards & Trend Indicators
- **Real-Time Key Metrics**: Total Revenue, Order Volume, Average Order Value (AOV), Units Sold, Active Customers, and Top Selling Product.
- **Period-over-Period Deltas**: Comparison trend badges (`▲ +X.X%` / `▼ -X.X%`) with color-coded status styling.

### 3. Interactive Plotly Visualizations
- **📈 Dual-Axis Revenue Trend**: Bar chart for Revenue and Line overlay for Order Count over time.
- **🗓️ Monthly Sales Heatmap**: Day-of-Week vs. Month sales intensity matrix.
- **🎯 Product Performance Matrix**: Price vs. Quantity scatter plot with Revenue bubble sizing.
- **🍩 Category Revenue Donut Chart**: Donut breakdown of sales distribution.
- **🌍 Geographic Regional Map**: Regional sales overview.
- **🏆 Salesperson Leaderboard**: Horizontal bar ranking top sales representatives.
- **📊 Time Series Decomposition**: Multi-panel chart isolating Trend, Seasonality, and Residual noise components.

### 4. AI & Machine Learning Insights Suite
- **🚨 Isolation Forest Anomaly Detection**: Unsupervised ML model identifying unusual transaction surges or volume drops.
- **🔮 30-Day Predictive Forecasting**: Holt-Winters Exponential Smoothing (with polynomial trend fallback) projecting future daily sales with 95% confidence bands.
- **💡 Smart Business Recommendations**: Automated insights flagging underperforming products, optimal promotional timing, and revenue concentration risks.
- **✨ Natural Language Executive Narrative**: Auto-generated executive brief for executive leadership.

### 5. Advanced Dynamic Filtering Panel
- **Date Range Presets**: Last 7 Days, This Month, Last Quarter, Year-to-Date, and Custom Date Pickers.
- **Multi-Select Dropdowns**: Category, Region, Salesperson, and Payment Method filters.
- **Dynamic Range Sliders**: Price Range and Order Quantity sliders.
- **Product Search Box**: Instant string matching search.
- **Reset & Counter Badge**: One-click reset button and active filter counter badge.

### 6. Multi-Format Export Studio
- **PDF Executive Report**: Formatted PDF document built via ReportLab containing summary metrics, KPI tables, and product leaderboards.
- **CSV Data Export**: One-click download of filtered dataset.

### 7. Modern Glassmorphism UI/UX
- Custom CSS design system featuring Thiranex Solutions branding, neon gradients, glassmorphism cards, responsive grids, onboarding tour guide, and FAQ section.

---

## 🛠️ Project Structure

```
Thiranex/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Technical documentation
├── assets/
│   ├── style.css              # Glassmorphism & dark theme stylesheet
│   └── logo.svg               # Thiranex Solutions vector logo asset
└── modules/
    ├── __init__.py
    ├── data_loader.py          # CSV/Excel/JSON loading & synthetic data generator
    ├── preprocessor.py        # Smart column auto-detection & data profiling
    ├── kpi_calculator.py      # Core metrics & MoM trend delta calculations
    ├── chart_builder.py       # 7 Plotly visualization builders
    ├── ai_insights.py         # Isolation Forest anomaly detection & forecasting
    ├── filters.py             # Sidebar & top header filter controls
    ├── exporters.py           # ReportLab PDF report generation & CSV export
    └── ui_components.py       # CSS injection, KPI HTML cards, header & onboarding tour
```

---

## 🚀 Installation & Running Locally

### 1. Prerequisites
Ensure **Python 3.9+** is installed on your system.

### 2. Install Dependencies
Open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
Run the main app file:
```bash
streamlit run app.py
```

The application will open automatically in your web browser at `http://localhost:8501`.

---

## 🧪 Verification & Code Validation

To verify all modules and run pre-flight syntax checks:
```bash
python -m py_compile app.py modules/*.py
```

---

## 📜 License & Ownership

© 2026 **Thiranex Solutions** — Built with ❤️ for Enterprise Analytics.
