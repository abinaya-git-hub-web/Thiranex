"""
Module: kpi_calculator.py
Description: Calculates core enterprise KPIs and period-over-period percentage trend indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_kpis(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Computes key performance indicators (KPIs) and trend deltas vs prior comparison period.

    Args:
        df (pd.DataFrame): Filtered sales DataFrame.
        mappings (Dict[str, Optional[str]]): Column role mappings.

    Returns:
        Dict[str, Any]: Dictionary containing current values, trend percentage changes, and badges.
    """
    date_col = mappings.get("date")
    rev_col = mappings.get("revenue")
    qty_col = mappings.get("quantity")
    product_col = mappings.get("product")
    cust_col = mappings.get("customer")

    if df.empty:
        return {
            "total_revenue": 0.0, "revenue_delta": 0.0,
            "total_orders": 0, "orders_delta": 0.0,
            "aov": 0.0, "aov_delta": 0.0,
            "units_sold": 0, "units_delta": 0.0,
            "active_customers": 0, "customers_delta": 0.0,
            "top_product": "N/A", "top_product_revenue": 0.0
        }

    # Extract Current Metrics
    total_revenue = float(df[rev_col].sum()) if rev_col and rev_col in df.columns else 0.0
    total_orders = len(df)
    aov = total_revenue / total_orders if total_orders > 0 else 0.0
    units_sold = int(df[qty_col].sum()) if qty_col and qty_col in df.columns else 0
    active_customers = int(df[cust_col].nunique()) if cust_col and cust_col in df.columns else total_orders

    # Determine Top Selling Product
    top_product = "N/A"
    top_product_revenue = 0.0
    if product_col and product_col in df.columns and rev_col and rev_col in df.columns:
        prod_grp = df.groupby(product_col)[rev_col].sum().sort_values(ascending=False)
        if not prod_grp.empty:
            top_product = str(prod_grp.index[0])
            top_product_revenue = float(prod_grp.iloc[0])

    # Compute Period-over-Period Deltas if Date Column exists
    revenue_delta = 0.0
    orders_delta = 0.0
    aov_delta = 0.0
    units_delta = 0.0
    customers_delta = 0.0

    if date_col and date_col in df.columns and len(df) > 1:
        df_sorted = df.sort_values(date_col)
        min_date = df_sorted[date_col].min()
        max_date = df_sorted[date_col].max()
        duration_days = (max_date - min_date).days

        if duration_days > 1:
            midpoint_date = min_date + pd.Timedelta(days=duration_days / 2)
            
            curr_df = df_sorted[df_sorted[date_col] >= midpoint_date]
            prev_df = df_sorted[df_sorted[date_col] < midpoint_date]

            if not prev_df.empty and not curr_df.empty:
                # Revenue Delta
                prev_rev = float(prev_df[rev_col].sum()) if rev_col else 0.0
                curr_rev = float(curr_df[rev_col].sum()) if rev_col else 0.0
                revenue_delta = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0.0

                # Orders Delta
                prev_orders = len(prev_df)
                curr_orders = len(curr_df)
                orders_delta = ((curr_orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0.0

                # AOV Delta
                prev_aov = prev_rev / prev_orders if prev_orders > 0 else 0.0
                curr_aov = curr_rev / curr_orders if curr_orders > 0 else 0.0
                aov_delta = ((curr_aov - prev_aov) / prev_aov * 100) if prev_aov > 0 else 0.0

                # Units Delta
                prev_units = float(prev_df[qty_col].sum()) if qty_col else 0.0
                curr_units = float(curr_df[qty_col].sum()) if qty_col else 0.0
                units_delta = ((curr_units - prev_units) / prev_units * 100) if prev_units > 0 else 0.0

                # Customers Delta
                prev_cust = float(prev_df[cust_col].nunique()) if cust_col else prev_orders
                curr_cust = float(curr_df[cust_col].nunique()) if cust_col else curr_orders
                customers_delta = ((curr_cust - prev_cust) / prev_cust * 100) if prev_cust > 0 else 0.0

    return {
        "total_revenue": total_revenue,
        "revenue_delta": np.round(revenue_delta, 1),
        "total_orders": total_orders,
        "orders_delta": np.round(orders_delta, 1),
        "aov": np.round(aov, 2),
        "aov_delta": np.round(aov_delta, 1),
        "units_sold": units_sold,
        "units_delta": np.round(units_delta, 1),
        "active_customers": active_customers,
        "customers_delta": np.round(customers_delta, 1),
        "top_product": top_product,
        "top_product_revenue": top_product_revenue
    }
