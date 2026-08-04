"""
Module: preprocessor.py
Description: Smart auto-detection of column roles, data cleaning, and dataset profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def detect_column_types(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently identifies semantic column roles (Date, Revenue, Quantity, Price, Product, etc.)
    using pattern matching on header names and column data types.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Optional[str]]: Dictionary mapping standard role names to actual column names.
    """
    column_mapping: Dict[str, Optional[str]] = {
        "date": None,
        "revenue": None,
        "quantity": None,
        "price": None,
        "product": None,
        "category": None,
        "region": None,
        "salesperson": None,
        "customer": None,
        "payment_method": None
    }

    cols = list(df.columns)
    cols_lower = {col: str(col).lower().strip().replace("_", " ").replace("-", " ") for col in cols}

    # 1. Detect Date Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["date", "time", "timestamp", "day", "month", "created at", "order date"]):
            column_mapping["date"] = col
            break

    if not column_mapping["date"]:
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                column_mapping["date"] = col
                break

    # 2. Detect Revenue Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["total revenue", "revenue", "total sales", "sales amount", "grand total", "net sales", "amount"]):
            column_mapping["revenue"] = col
            break

    # 3. Detect Quantity Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["quantity", "qty", "units sold", "units", "volume", "items count"]):
            column_mapping["quantity"] = col
            break

    # 4. Detect Unit Price Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["unit price", "price", "rate", "unit cost", "cost per unit"]):
            column_mapping["price"] = col
            break

    # 5. Detect Product Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["product", "item", "sku", "product name", "item name", "service"]):
            column_mapping["product"] = col
            break

    # 6. Detect Category Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["category", "segment", "department", "product category", "group", "type"]):
            column_mapping["category"] = col
            break

    # 7. Detect Region Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["region", "country", "state", "territory", "location", "zone", "city"]):
            column_mapping["region"] = col
            break

    # 8. Detect Salesperson Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["salesperson", "sales rep", "rep", "agent", "account owner", "owner", "seller"]):
            column_mapping["salesperson"] = col
            break

    # 9. Detect Customer Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["customer", "client", "customer id", "client name", "account id", "user id"]):
            column_mapping["customer"] = col
            break

    # 10. Detect Payment Method Column
    for col, lower in cols_lower.items():
        if any(term in lower for term in ["payment", "payment method", "pay mode", "payment type", "transaction type"]):
            column_mapping["payment_method"] = col
            break

    return column_mapping


def standardize_and_clean_data(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Standardizes data types, parses date formats, calculates derived columns if missing,
    and strips extra whitespace from string fields.

    Args:
        df (pd.DataFrame): Raw uploaded DataFrame.
        mappings (Dict[str, Optional[str]]): Auto-detected column mappings.

    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame.
    """
    cleaned_df = df.copy()

    # Parse dates
    date_col = mappings.get("date")
    if date_col and date_col in cleaned_df.columns:
        cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors="coerce")
        # Drop rows where date failed to parse
        cleaned_df = cleaned_df.dropna(subset=[date_col])

    # Convert numeric columns safely
    for key in ["revenue", "quantity", "price"]:
        col = mappings.get(key)
        if col and col in cleaned_df.columns:
            if cleaned_df[col].dtype == object:
                # Remove currency symbols ($ , € £ ₹)
                cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce").fillna(0)

    # Compute Revenue if missing but Quantity & Price exist
    rev_col = mappings.get("revenue")
    qty_col = mappings.get("quantity")
    price_col = mappings.get("price")

    if (not rev_col or rev_col not in cleaned_df.columns) and (qty_col and price_col):
        cleaned_df["Calculated_Revenue"] = cleaned_df[qty_col] * cleaned_df[price_col]
        mappings["revenue"] = "Calculated_Revenue"

    # Clean categorical string columns
    for key in ["product", "category", "region", "salesperson", "payment_method", "customer"]:
        col = mappings.get(key)
        if col and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    return cleaned_df


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates dataset profiling summary statistics (missing values, data types, unique counts).

    Args:
        df (pd.DataFrame): DataFrame to profile.

    Returns:
        Dict[str, Any]: Profiling metadata summary dictionary.
    """
    null_counts = df.isnull().sum()
    null_percentages = np.round((null_counts / len(df)) * 100, 2)

    profile_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Missing Values": null_counts,
        "Missing (%)": null_percentages,
        "Unique Values": df.nunique()
    })

    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = round(memory_bytes / (1024 * 1024), 2)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_mb": memory_mb,
        "column_profile": profile_df,
        "numeric_summary": df.describe(include=[np.number]).T if not df.select_dtypes(include=[np.number]).empty else None
    }
