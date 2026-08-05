"""
Module: data_loader.py
Description: Handles synthetic customer dataset generation with 3 realistic segments 
             and custom file uploads (CSV, Excel, JSON) for Thiranex Solutions.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from datetime import datetime, timedelta


@st.cache_data(show_spinner=False)
def generate_sample_data(num_rows: int = 1000) -> pd.DataFrame:
    """
    Generates 1000+ realistic synthetic customer records with 3 distinct behavioral segments:
    1. High-Value Premium (20%): High spend, high frequency, high income
    2. Regular Bargain (35%): Medium spend, medium frequency, price-sensitive
    3. Occasional Explorers (45%): Low spend, low frequency, exploring products

    Returns:
        pd.DataFrame: Comprehensive customer dataset ready for segmentation & ML analysis.
    """
    np.random.seed(42)

    # Segment allocation ratios
    n_premium = int(num_rows * 0.20)
    n_regular = int(num_rows * 0.35)
    n_occasional = num_rows - n_premium - n_regular

    categories_pool = ["Electronics", "Clothing", "Grocery", "Home", "Beauty", "Sports", "Books", "Toys"]
    payment_pool = ["Credit Card", "Debit Card", "UPI", "Cash", "Net Banking"]
    occupations_pool = ["Professional", "Student", "Retired", "Business", "Freelancer", "Homemaker"]
    locations_pool = ["North", "South", "East", "West", "Central"]
    genders_pool = ["Male", "Female", "Other"]

    today = datetime.now()

    # --- 1. High-Value Premium Segment (20%) ---
    premium_ids = [f"CUST-{i+1:04d}" for i in range(n_premium)]
    premium_age = np.random.randint(28, 62, size=n_premium)
    premium_gender = np.random.choice(genders_pool, size=n_premium, p=[0.48, 0.48, 0.04])
    premium_location = np.random.choice(locations_pool, size=n_premium, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    premium_income = np.random.randint(90000, 150001, size=n_premium)
    premium_occ = np.random.choice(["Professional", "Business", "Freelancer"], size=n_premium, p=[0.60, 0.30, 0.10])
    premium_spend = np.random.uniform(5500, 10000, size=n_premium)
    premium_freq = np.random.randint(28, 51, size=n_premium)
    premium_aov = np.round(premium_spend / premium_freq, 2)
    premium_cats = [", ".join(np.random.choice(categories_pool, size=np.random.randint(3, 6), replace=False)) for _ in range(n_premium)]
    premium_last_days = np.random.randint(1, 45, size=n_premium)
    premium_since_days = np.random.randint(365, 1095, size=n_premium)
    premium_pay = np.random.choice(payment_pool, size=n_premium, p=[0.55, 0.15, 0.20, 0.02, 0.08])
    premium_sat = np.random.choice([4, 5], size=n_premium, p=[0.30, 0.70])

    # --- 2. Regular Bargain Segment (35%) ---
    start_idx = n_premium
    regular_ids = [f"CUST-{start_idx + i + 1:04d}" for i in range(n_regular)]
    regular_age = np.random.randint(22, 65, size=n_regular)
    regular_gender = np.random.choice(genders_pool, size=n_regular, p=[0.47, 0.49, 0.04])
    regular_location = np.random.choice(locations_pool, size=n_regular, p=[0.20, 0.25, 0.25, 0.15, 0.15])
    regular_income = np.random.randint(45000, 90000, size=n_regular)
    regular_occ = np.random.choice(occupations_pool, size=n_regular, p=[0.35, 0.15, 0.10, 0.15, 0.15, 0.10])
    regular_spend = np.random.uniform(1800, 5499, size=n_regular)
    regular_freq = np.random.randint(12, 28, size=n_regular)
    regular_aov = np.round(regular_spend / regular_freq, 2)
    regular_cats = [", ".join(np.random.choice(categories_pool, size=np.random.randint(2, 4), replace=False)) for _ in range(n_regular)]
    regular_last_days = np.random.randint(10, 120, size=n_regular)
    regular_since_days = np.random.randint(180, 1000, size=n_regular)
    regular_pay = np.random.choice(payment_pool, size=n_regular, p=[0.25, 0.35, 0.25, 0.05, 0.10])
    regular_sat = np.random.choice([3, 4, 5], size=n_regular, p=[0.25, 0.55, 0.20])

    # --- 3. Occasional Explorers Segment (45%) ---
    start_idx = n_premium + n_regular
    occ_ids = [f"CUST-{start_idx + i + 1:04d}" for i in range(n_occasional)]
    occ_age = np.random.randint(18, 70, size=n_occasional)
    occ_gender = np.random.choice(genders_pool, size=n_occasional, p=[0.46, 0.50, 0.04])
    occ_location = np.random.choice(locations_pool, size=n_occasional, p=[0.18, 0.22, 0.20, 0.22, 0.18])
    occ_income = np.random.randint(20000, 50000, size=n_occasional)
    occ_occ = np.random.choice(occupations_pool, size=n_occasional, p=[0.20, 0.35, 0.15, 0.05, 0.15, 0.10])
    occ_spend = np.random.uniform(100, 1799, size=n_occasional)
    occ_freq = np.random.randint(1, 12, size=n_occasional)
    occ_aov = np.round(occ_spend / occ_freq, 2)
    occ_cats = [", ".join(np.random.choice(categories_pool, size=np.random.randint(1, 3), replace=False)) for _ in range(n_occasional)]
    occ_last_days = np.random.randint(30, 365, size=n_occasional)
    occ_since_days = np.random.randint(30, 730, size=n_occasional)
    occ_pay = np.random.choice(payment_pool, size=n_occasional, p=[0.15, 0.30, 0.40, 0.10, 0.05])
    occ_sat = np.random.choice([1, 2, 3, 4], size=n_occasional, p=[0.15, 0.25, 0.40, 0.20])

    # Combine lists
    cust_ids = premium_ids + regular_ids + occ_ids
    ages = np.concatenate([premium_age, regular_age, occ_age])
    genders = np.concatenate([premium_gender, regular_gender, occ_gender])
    locations = np.concatenate([premium_location, regular_location, occ_location])
    incomes = np.concatenate([premium_income, regular_income, occ_income])
    occupations = np.concatenate([premium_occ, regular_occ, occ_occ])
    spends = np.round(np.concatenate([premium_spend, regular_spend, occ_spend]), 2)
    freqs = np.concatenate([premium_freq, regular_freq, occ_freq])
    aovs = np.round(np.concatenate([premium_aov, regular_aov, occ_aov]), 2)
    cats = premium_cats + regular_cats + occ_cats
    last_days = np.concatenate([premium_last_days, regular_last_days, occ_last_days])
    since_days = np.concatenate([premium_since_days, regular_since_days, occ_since_days])
    payments = np.concatenate([premium_pay, regular_pay, occ_pay])
    satisfactions = np.concatenate([premium_sat, regular_sat, occ_sat])

    last_purchase_dates = [(today - timedelta(days=int(d))).strftime("%Y-%m-%d") for d in last_days]
    customer_since_dates = [(today - timedelta(days=int(d))).strftime("%Y-%m-%d") for d in since_days]
    order_counts = freqs  # Equivalent to purchase frequency

    df = pd.DataFrame({
        "Customer ID": cust_ids,
        "Age": ages,
        "Gender": genders,
        "Location": locations,
        "Income": incomes,
        "Occupation": occupations,
        "Total Spend ($)": spends,
        "Purchase Frequency": freqs,
        "Order Count": order_counts,
        "Average Order Value ($)": aovs,
        "Product Categories Purchased": cats,
        "Last Purchase Date": last_purchase_dates,
        "Customer Since": customer_since_dates,
        "Preferred Payment": payments,
        "Satisfaction Score": satisfactions
    })

    # Shuffle rows to avoid contiguous order
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parses uploaded CSV, Excel, or JSON file into pandas DataFrame.
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
            return None, "The uploaded dataset is empty."

        if len(df) > 50000:
            st.info(f"⚡ Dataset contains {len(df):,} rows. Downsampling 50,000 rows for optimal interactivity.")
            df = df.sample(n=50000, random_state=42).sort_index()

        return df, None

    except Exception as e:
        return None, f"Failed to parse dataset: {str(e)}"
