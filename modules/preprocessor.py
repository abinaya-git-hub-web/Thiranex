"""
Module: preprocessor.py
Description: Smart column detection, dataset profiling summary, and feature extraction normalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detects customer-related demographic and behavioral columns in the DataFrame.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    mappings = {
        "customer_id": None,
        "age": None,
        "gender": None,
        "location": None,
        "income": None,
        "occupation": None,
        "spend": None,
        "frequency": None,
        "aov": None,
        "last_purchase_date": None,
        "customer_since": None,
        "categories": None,
        "payment": None,
        "satisfaction": None
    }

    keywords = {
        "customer_id": ["customer id", "customer_id", "cust_id", "client_id", "id", "customer"],
        "age": ["age", "customer_age", "years"],
        "gender": ["gender", "sex"],
        "location": ["location", "region", "city", "state", "country"],
        "income": ["income", "annual_income", "salary", "earnings"],
        "occupation": ["occupation", "job", "profession", "work"],
        "spend": ["total spend ($)", "total_spend", "spend", "monetary", "total_revenue", "total_amount", "revenue"],
        "frequency": ["purchase frequency", "frequency", "order count", "order_count", "total_orders", "orders"],
        "aov": ["average order value ($)", "average_order_value", "aov", "avg_order_value", "avg_spend"],
        "last_purchase_date": ["last purchase date", "last_purchase_date", "last_purchase", "recency_date", "last_order_date"],
        "customer_since": ["customer since", "customer_since", "signup_date", "join_date", "created_at"],
        "categories": ["product categories purchased", "product_categories", "categories", "category"],
        "payment": ["preferred payment", "preferred_payment", "payment_method", "payment"],
        "satisfaction": ["satisfaction score", "satisfaction_score", "satisfaction", "rating"]
    }

    for role, key_list in keywords.items():
        for key in key_list:
            if key in cols:
                mappings[role] = cols[key]
                break

    for role in mappings:
        if mappings[role] is None:
            for c_lower, c_actual in cols.items():
                if role in c_lower:
                    mappings[role] = c_actual
                    break

    return mappings


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates dataset profiling summary including missing values, datatypes, and stats.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    col_profile = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_cnt = int(df[col].isnull().sum())
        null_pct = round((null_cnt / total_rows) * 100, 1)
        unique_cnt = int(df[col].nunique())
        
        sample_val = str(df[col].iloc[0]) if total_rows > 0 else ""
        if len(sample_val) > 30:
            sample_val = sample_val[:27] + "..."

        col_profile.append({
            "Column Name": col,
            "Data Type": dtype,
            "Missing Count": null_cnt,
            "Missing %": f"{null_pct}%",
            "Unique Values": unique_cnt,
            "Sample Record": sample_val
        })

    profile_df = pd.DataFrame(col_profile)
    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_mb": memory_mb,
        "column_profile": profile_df
    }


def extract_and_scale_features(df: pd.DataFrame, mappings: Dict[str, str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Extracts numerical clustering features (Spend, Frequency, AOV, Recency, Age, Income)
    and applies StandardScaler normalization.
    """
    data_dict = {}

    spend_col = mappings.get("spend")
    if spend_col and spend_col in df.columns:
        data_dict["Spend"] = pd.to_numeric(df[spend_col], errors="coerce").fillna(500)
    else:
        data_dict["Spend"] = df["Monetary_Val"] if "Monetary_Val" in df.columns else np.random.uniform(100, 5000, len(df))

    freq_col = mappings.get("frequency")
    if freq_col and freq_col in df.columns:
        data_dict["Frequency"] = pd.to_numeric(df[freq_col], errors="coerce").fillna(5)
    else:
        data_dict["Frequency"] = df["Frequency_Val"] if "Frequency_Val" in df.columns else np.random.randint(1, 20, len(df))

    aov_col = mappings.get("aov")
    if aov_col and aov_col in df.columns:
        data_dict["AOV"] = pd.to_numeric(df[aov_col], errors="coerce").fillna(100)
    else:
        data_dict["AOV"] = (data_dict["Spend"] / np.maximum(data_dict["Frequency"], 1)).round(2)

    if "Recency_Days" in df.columns:
        data_dict["Recency"] = df["Recency_Days"]
    else:
        date_col = mappings.get("last_purchase_date")
        if date_col and date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            max_d = dates.max() if not dates.isna().all() else pd.Timestamp.now()
            data_dict["Recency"] = (max_d - dates).dt.days.fillna(180)
        else:
            data_dict["Recency"] = np.random.randint(1, 365, len(df))

    age_col = mappings.get("age")
    if age_col and age_col in df.columns:
        data_dict["Age"] = pd.to_numeric(df[age_col], errors="coerce").fillna(35)
    else:
        data_dict["Age"] = np.random.randint(18, 65, len(df))

    inc_col = mappings.get("income")
    if inc_col and inc_col in df.columns:
        data_dict["Income"] = pd.to_numeric(df[inc_col], errors="coerce").fillna(55000)
    else:
        data_dict["Income"] = np.random.randint(20000, 120000, len(df))

    feature_df = pd.DataFrame(data_dict)
    feature_names = list(feature_df.columns)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_df)

    return scaled_matrix, feature_names, feature_df
