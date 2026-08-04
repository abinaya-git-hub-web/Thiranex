"""
Module: data_loader.py
Description: Handles file loading for CSV, Excel, and JSON formats, as well as 
             synthetic sample data generation for Thiranex Solutions.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional


@st.cache_data(show_spinner=False)
def generate_sample_data(num_rows: int = 500) -> pd.DataFrame:
    """
    Generates realistic, synthetic sales transaction dataset for Thiranex Solutions.

    Args:
        num_rows (int): Number of synthetic sales records to generate. Default 500.

    Returns:
        pd.DataFrame: Structured sales data with dates, products, categories, 
                     regions, salespersons, quantities, prices, revenue, and payment methods.
    """
    np.random.seed(42)

    # Date range across 12 full months (Jan 2025 to Dec 2025)
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = np.random.choice(date_range, size=num_rows)
    dates.sort()

    # Product catalog structure: (Product Name, Category, Base Price)
    catalog = [
        ("Cloud Analytics Suite", "Enterprise Software", 1200.0),
        ("CyberGuard Shield Pro", "Enterprise Software", 850.0),
        ("AI Predictive Insights Engine", "Enterprise Software", 2100.0),
        ("Thiranex IoT Gateway X1", "Hardware & Sensors", 450.0),
        ("Smart Sensor Hub Enterprise", "Hardware & Sensors", 320.0),
        ("Edge Computing Station", "Hardware & Sensors", 1600.0),
        ("Managed Cloud Operations", "Consulting Services", 3500.0),
        ("DevOps Migration Package", "Consulting Services", 2800.0),
        ("Security Audit & Compliance", "Consulting Services", 1950.0),
        ("24/7 Priority SLA Support", "Consulting Services", 900.0),
    ]

    selected_catalog_indices = np.random.choice(len(catalog), size=num_rows)
    
    products = [catalog[i][0] for i in selected_catalog_indices]
    categories = [catalog[i][1] for i in selected_catalog_indices]
    base_prices = np.array([catalog[i][2] for i in selected_catalog_indices])

    # Dynamic pricing variability (+/- 10%)
    price_multipliers = np.random.uniform(0.90, 1.10, size=num_rows)
    unit_prices = np.round(base_prices * price_multipliers, 2)

    # Order quantities (1 to 15 units)
    quantities = np.random.randint(1, 16, size=num_rows)
    revenue = np.round(unit_prices * quantities, 2)

    # Demographic dimensions
    regions = np.random.choice(["North", "South", "East", "West", "Central"], size=num_rows, p=[0.25, 0.25, 0.20, 0.20, 0.10])
    salespersons = np.random.choice(["Aarav Sharma", "Priya Patel", "Vikram Malhotra", "Ananya Reddy", "Rahul Verma"], size=num_rows)
    payment_methods = np.random.choice(["Credit Card", "UPI", "Bank Wire", "Corporate Net Banking"], size=num_rows, p=[0.35, 0.35, 0.15, 0.15])
    
    customer_ids = [f"CUST-{np.random.randint(1000, 9999)}" for _ in range(num_rows)]
    order_ids = [f"TX-ORD-{10000 + i}" for i in range(num_rows)]

    df = pd.DataFrame({
        "Order ID": order_ids,
        "Date": dates,
        "Customer ID": customer_ids,
        "Product": products,
        "Category": categories,
        "Region": regions,
        "Salesperson": salespersons,
        "Quantity": quantities,
        "Unit Price ($)": unit_prices,
        "Total Revenue ($)": revenue,
        "Payment Method": payment_methods
    })

    # Introduce minor deliberate anomalies for AI Isolation Forest to detect
    anomaly_indices = [15, 88, 230, 412]
    for idx in anomaly_indices:
        df.at[idx, "Total Revenue ($)"] = df.at[idx, "Total Revenue ($)"] * 6.5
        df.at[idx, "Quantity"] = df.at[idx, "Quantity"] * 5

    return df


@st.cache_data(show_spinner=False)
def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parses user uploaded CSV, Excel, or JSON files into a Pandas DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: Loaded DataFrame and error message if any.
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            return None, f"Unsupported file format: {uploaded_file.name}. Please upload CSV, Excel, or JSON."

        if df.empty:
            return None, "The uploaded file is empty."

        # Performance Optimization: Sampling datasets > 50,000 rows
        if len(df) > 50000:
            st.info(f"⚡ Large dataset detected ({len(df):,} rows). Sampling 50,000 rows for optimal responsiveness.")
            df = df.sample(n=50000, random_state=42).sort_index()

        return df, None

    except Exception as e:
        return None, f"Failed to parse dataset: {str(e)}"
