# Thiranex Solutions — Customer Segmentation & Persona Intelligence Application

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)

An enterprise-grade Customer Segmentation Application built for **Thiranex Solutions**. The platform utilizes machine learning algorithms (K-Means, DBSCAN, Hierarchical Agglomerative Clustering) alongside traditional RFM Analysis, Random Forest Churn Prediction, Customer Lifetime Value (CLV) Forecasting, and Automated Persona Building.

---

## 📁 Repository & Codebase Structure

```
Thiranex_Customer_Segmentation/
├── app.py                      # Main entry point & Streamlit app
├── requirements.txt            # Python package dependencies
├── README.md                   # Application documentation
├── assets/
│   ├── style.css               # Dark theme glassmorphism CSS
│   └── logo.svg                # Thiranex Solutions brand SVG logo
└── modules/
    ├── __init__.py             # Modules package initializer
    ├── data_loader.py          # File uploader & 1,000 synthetic customer generator
    ├── preprocessor.py         # Smart column detection & feature normalization
    ├── rfm_analyzer.py         # RFM scoring (1-5 scale) & segment classification
    ├── kmeans_cluster.py       # K-Means clustering with Elbow Method & Silhouette Score
    ├── dbscan_cluster.py       # DBSCAN density-based outlier detection
    ├── hierarchical_cluster.py # Agglomerative clustering & Scipy linkage matrix
    ├── persona_builder.py      # Customer persona cards & natural language narrative
    ├── chart_builder.py        # 12+ Plotly interactive dark-themed charts
    ├── ai_predictor.py         # Random Forest Churn Classifier, CLV & Next Best Offer
    ├── filters.py              # Interactive sidebar filters & customer search
    ├── exporters.py            # PDF executive report builder & CSV exporter
    └── ui_components.py        # Glassmorphic cards, KPI renderers, header & footer
```

---

## ⚡ Key Features

1. **Synthetic Data Generator**: 1,000 synthetic customer records with 3 natural segments (*High-Value Premium*, *Regular Bargain*, *Occasional Explorers*).
2. **Multi-Format Upload Studio**: Parses CSV, Excel (`.xlsx`, `.xls`), and JSON files with smart downsampling.
3. **Smart Column Auto-Detection**: Detects demographic and behavioral columns automatically.
4. **4 Segmentation Methods**:
   - **K-Means Clustering (ML)**
   - **RFM Analysis (Traditional)**
   - **DBSCAN Clustering (Outliers)**
   - **Hierarchical Clustering (Agglomerative)**
5. **Customer Persona Builder**: Auto-generates demographic, monetary, and behavioral profiles, marketing strategies, recommended offers, and churn risk badges.
6. **Advanced AI Features**:
   - **Random Forest Churn Prediction & Driver Importances**
   - **Customer Lifetime Value (CLV) Regression & Tiers**
   - **Next Best Offer Recommendation Engine**
7. **10+ Plotly Visualizations**: Donut chart, Radar chart, Demographic heatmap, Spending box plots, 3D RFM scatter plot, Lifecycle Sankey flow, Scipy Dendrogram tree, Elbow curve, Silhouette evaluation, Churn drivers, and CLV tiers.
8. **Export Studio**: Generates executive PDF reports via ReportLab and CSV exports.

---

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launching the App

Run Streamlit:

```bash
streamlit run app.py
```

---

## 🛡️ License & Credits

© 2026 **Thiranex Solutions** — Enterprise Machine Learning Intelligence Studio. Built with ❤️.
