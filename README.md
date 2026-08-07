# Thiranex Solutions — Enterprise Data Cleaning & Reporting Automation Platform

A complete, production-ready ETL (Extract-Transform-Load), automated data profiling, intelligent data cleaning, and multi-format reporting engine built for **Thiranex Solutions** using Python, Streamlit, Pandas, Scikit-learn, OpenPyXL, ReportLab, and Plotly.

---

## 📁 Project Architecture & Modular Structure

```text
Thiranex_Data_Cleaning_Automation/
├── app.py                      # Main entry point
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── assets/
│   ├── style.css               # Custom design
│   └── logo.svg                # Brand logo
└── modules/
    ├── __init__.py             # Module package exports
    ├── data_loader.py          # Multi-source data ingestion (CSV, Excel, JSON, XML, Parquet, SQL, MongoDB, Cloud, API)
    ├── data_profiler.py        # Data quality assessment & Health Scorecard (0-100)
    ├── missing_handler.py      # Missing value strategies (Mean, Median, Mode, Forward/Backward Fill, Interpolation, KNN)
    ├── duplicate_handler.py    # Exact & Fuzzy duplicate detection/removal (Levenshtein)
    ├── inconsistency_handler.py # Standardization (Casing, dates, phone, units) & validation
    ├── outlier_detector.py     # Outlier detection/handling (IQR, Z-Score, Isolation Forest)
    ├── feature_engineer.py     # Feature engineering, scaling (Min-Max, Standard), encoding (One-Hot, Label)
    ├── data_repair.py          # ML-based imputation (Random Forest)
    ├── pipeline_builder.py     # Visual workflow builder & JSON preset serialization
    ├── report_generator.py     # Multi-format reports (PDF, Excel, HTML, JSON)
    ├── scheduler.py            # Automated job scheduling & directory watcher
    ├── chart_builder.py        # 12+ Plotly visualization functions
    ├── filters.py              # Interactive dataset filters
    ├── exporters.py            # PDF, CSV, Excel, Parquet export functions
    └── ui_components.py        # UI components, header banner, glassmorphic cards
```

---

## ⚡ Core Features & Capabilities

### 1. Data Import & Multi-Source Integration (`modules/data_loader.py`)
- **File Formats**: CSV, Excel (`.xlsx`, `.xls`), JSON, XML, Parquet, Feather.
- **Batch Upload**: Multi-file drag & drop with automated schema concatenation.
- **Database Connectivity**: SQL (`SQLite`, `MySQL`, `PostgreSQL`) & NoSQL (`MongoDB`) connectors with local simulation fallback.
- **Cloud Storage**: Ingest from Google Sheets API, AWS S3, and Azure Blob Storage.
- **REST API & Web Ingestion**: Live JSON endpoint parser.
- **Synthetic Messy Data Generator**: Instantly generate enterprise CRM, Financial, or HR datasets injected with missing values, fuzzy duplicates, unformatted phones, mixed date formats, and extreme numeric outliers.

### 2. Automated Data Profiling (`modules/data_profiler.py`)
- **Structural Assessment**: Row/col counts, memory footprint, data types.
- **Enterprise Data Quality Scorecard (0-100)**: Calculates overall Data Health Score along 4 dimensions (Completeness, Accuracy, Consistency, Timeliness) with color-coded status badges (**Good**, **Warning**, **Critical**).
- **Statistical Summary**: Skewness, kurtosis, IQR, median, mode, standard deviation.
- **Regex Pattern Detection**: Detects emails, phone numbers, zip codes, IP addresses, URLs, and date formats.

### 3. Intelligent Data Cleaning Engine (`modules/missing_handler.py`, `modules/duplicate_handler.py`, `modules/inconsistency_handler.py`, `modules/outlier_detector.py`, `modules/data_repair.py`)
- **Missing Value Handling**: Drop rows/cols, Mean/Median/Mode, Forward/Backward Fill, Linear Interpolation, **KNN Imputation**, and **Random Forest ML Imputation**.
- **Fuzzy Record Linkage & Deduplication**: Purges exact duplicate rows or fuzzy text duplicates using Levenshtein similarity with adjustable threshold sliders.
- **Text & Date Standardization**: Converts text casing (Title, Upper, Lower), standardizes phone formatting, converts units (kg/lbs), and parses dates to ISO `YYYY-MM-DD`.
- **Numeric Outlier Handling**: Detects outliers via IQR, Z-Score, or Isolation Forest, applying Winsorizing capping, removal, or log transformations.

### 4. Visual Pipeline Builder & Automation Engine (`modules/pipeline_builder.py`, `modules/scheduler.py`)
- **Interactive Sequence Builder**: Enable, reorder, or configure ETL cleaning steps.
- **Pipeline Templates**: Export and import pipeline configs as reusable JSON presets.
- **Automated Scheduler**: Define daily, weekly, or monthly scheduled cleaning jobs and watch directories for uncleaned files.

### 5. 12+ Interactive Plotly Visualizations (`modules/chart_builder.py`)
1. Data Quality Scorecard Gauges
2. Missing Value Heatmap (Before vs After)
3. Duplicate Record Ratio Pie Chart
4. Outlier Numeric Boxplots
5. Distribution Histograms with Density Curves
6. Data Type Donut Chart
7. Cleaning Impact Comparison Bar Chart
8. Pairwise Correlation Heatmap
9. Regex Pattern Frequency Chart
10. Row Volume Lineage Trend
11. Anomaly Detection Scatter Plot
12. ETL Pipeline Lineage Flow Diagram

### 6. Multi-Format Automated Reporting & Export (`modules/report_generator.py`, `modules/exporters.py`)
- **PDF Executive Briefing**: ReportLab PDF with Thiranex branding, scorecard KPI tables, executive summary, and audit log.
- **Excel Multi-Tab Workbook**: OpenPyXL workbook containing Executive Summary, Cleaned Data, Column Profile, Audit Log, and Data Dictionary tabs.
- **Interactive HTML Report**: Standalone web report with glassmorphism CSS & embedded tables.
- **Cleaned Data Downloads**: Download in CSV, Excel, JSON, or Parquet formats.

---

## 🛠️ Quickstart Installation & Execution

```bash
# 1. Navigate to project root
cd Thiranex

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

---

© 2026 **Thiranex Solutions** — Built with ❤️ for Production-Ready Enterprise ETL & Automation.
