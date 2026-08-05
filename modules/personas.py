"""
Module: personas.py
Description: Customer Persona Builder & Automated Natural Language Executive Narrative Engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


CREATIVE_NAME_MAP = {
    "High-Value Premium": "👑 Elite High-Rollers",
    "Regular Bargain": "🏷️ Smart Bargain Hunters",
    "Occasional Explorers": "🧭 Casual Digital Explorers",
    "Champions": "🏆 Brand Champions",
    "Loyal Customers": "💎 Loyal Mainstays",
    "Potential Loyalists": "🌱 Rising Stars",
    "New Customers": "✨ Fresh Converts",
    "At Risk": "⚠️ At-Risk Veterans",
    "Can't Lose": "🚨 High-Value Slipping",
    "Hibernating": "😴 Dormant Buyers",
    "Lost": "🌧️ Lost Customers",
    "Noise / Outliers": "⚡ Anomalous Power Buyers"
}

ACCENT_COLORS = ["#8B5CF6", "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#06B6D4", "#6366F1"]


def generate_segment_personas(df: pd.DataFrame, segment_col: str, mappings: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Generates rich, actionable Customer Persona profiles for each segment in the dataset.
    """
    if segment_col not in df.columns:
        return []

    if mappings is None:
        mappings = {}

    segments = df[segment_col].unique()
    personas = []

    total_customers = len(df)
    total_revenue_overall = df["Total Spend ($)"].sum() if "Total Spend ($)" in df.columns else df["Monetary_Val"].sum() if "Monetary_Val" in df.columns else 1.0

    for idx, seg in enumerate(sorted(segments, key=lambda x: str(x))):
        seg_df = df[df[segment_col] == seg]
        count = len(seg_df)
        pct = round((count / total_customers) * 100, 1)

        # Monetary stats
        spend_col = mappings.get("spend") if mappings.get("spend") in df.columns else ("Total Spend ($)" if "Total Spend ($)" in df.columns else "Monetary_Val")
        total_spend = seg_df[spend_col].sum() if spend_col in seg_df.columns else 0.0
        avg_spend = seg_df[spend_col].mean() if spend_col in seg_df.columns else 0.0
        rev_share = round((total_spend / total_revenue_overall) * 100, 1) if total_revenue_overall > 0 else 0.0

        # Frequency & AOV
        freq_col = mappings.get("frequency") if mappings.get("frequency") in df.columns else ("Purchase Frequency" if "Purchase Frequency" in df.columns else "Frequency_Val")
        avg_freq = round(seg_df[freq_col].mean(), 1) if freq_col in seg_df.columns else 1.0

        aov_col = mappings.get("aov") if mappings.get("aov") in df.columns else ("Average Order Value ($)" if "Average Order Value ($)" in df.columns else "AOV")
        avg_aov = round(seg_df[aov_col].mean(), 2) if aov_col in seg_df.columns else round(avg_spend / max(avg_freq, 1), 2)

        # Demographics
        age_col = mappings.get("age") if mappings.get("age") in df.columns else "Age"
        avg_age = int(round(seg_df[age_col].mean())) if age_col in seg_df.columns else 35
        min_age = int(seg_df[age_col].min()) if age_col in seg_df.columns else 20
        max_age = int(seg_df[age_col].max()) if age_col in seg_df.columns else 60

        inc_col = mappings.get("income") if mappings.get("income") in df.columns else "Income"
        avg_inc = seg_df[inc_col].mean() if inc_col in seg_df.columns else 50000.0

        loc_col = mappings.get("location") if mappings.get("location") in df.columns else "Location"
        top_loc = seg_df[loc_col].mode().iloc[0] if loc_col in seg_df.columns and not seg_df[loc_col].empty else "Central"

        occ_col = mappings.get("occupation") if mappings.get("occupation") in df.columns else "Occupation"
        top_occ = seg_df[occ_col].mode().iloc[0] if occ_col in seg_df.columns and not seg_df[occ_col].empty else "Professional"

        pay_col = mappings.get("payment") if mappings.get("payment") in df.columns else "Preferred Payment"
        top_pay = seg_df[pay_col].mode().iloc[0] if pay_col in seg_df.columns and not seg_df[pay_col].empty else "Credit Card"

        # Categories
        cat_col = mappings.get("categories") if mappings.get("categories") in df.columns else "Product Categories Purchased"
        top_cats_str = "Electronics, Home"
        if cat_col in seg_df.columns:
            cat_series = seg_df[cat_col].dropna().astype(str).str.split(", ").explode()
            if not cat_series.empty:
                top_cats_str = ", ".join(cat_series.value_counts().head(3).index.tolist())

        # Persona Naming & Strategy Logic
        raw_name = str(seg)
        creative_name = CREATIVE_NAME_MAP.get(raw_name, f"🎯 Segment {raw_name}")

        if avg_spend > 5000 or "Champion" in raw_name or "High-Value" in raw_name:
            strategy = "VIP White-Glove Concierge, early access to flagship products, multi-tier loyalty perks."
            offers = "Exclusive 25% Off Flagship Electronics + Dedicated Account Support"
            risk_level = "Low"
            risk_badge = "🟢 LOW RISK"
        elif avg_spend > 2000 or "Bargain" in raw_name or "Loyal" in raw_name:
            strategy = "Bundle discounts, tiered threshold rewards ($50 off $250), personalized email recommendations."
            offers = "Buy 2 Get 1 Free on Apparel & Free Shipping on Orders > $100"
            risk_level = "Medium"
            risk_badge = "🟡 MEDIUM RISK"
        elif "At Risk" in raw_name or "Can't Lose" in raw_name:
            strategy = "Aggressive win-back campaigns, feedback surveys with instant coupon incentives."
            offers = "We Miss You! 30% Off Your Next Order within 14 Days"
            risk_level = "High"
            risk_badge = "🔴 HIGH RISK"
        else:
            strategy = "Nurture with introductory guides, first-time buyer perks, product recommendations."
            offers = "15% Welcome Discount on First Category Cross-over Purchase"
            risk_level = "Medium"
            risk_badge = "🟡 MEDIUM RISK"

        color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]

        personas.append({
            "segment_id": raw_name,
            "creative_name": creative_name,
            "count": count,
            "percentage": pct,
            "revenue_share": rev_share,
            "avg_spend": avg_spend,
            "avg_aov": avg_aov,
            "avg_frequency": avg_freq,
            "demographics": {
                "age_range": f"{min_age} - {max_age} yrs (Avg: {avg_age})",
                "avg_income": f"${avg_inc:,.0f}",
                "top_location": top_loc,
                "top_occupation": top_occ
            },
            "behavioral": {
                "top_categories": top_cats_str,
                "preferred_payment": top_pay
            },
            "strategy": strategy,
            "recommended_offers": offers,
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "theme_color": color
        })

    return personas


def generate_automated_narrative(df: pd.DataFrame, segment_col: str, personas: List[Dict[str, Any]]) -> str:
    """
    Generates automated executive business insights summarizing segment breakdown and strategic roadmap.
    """
    if not personas:
        return "No segment data available to construct executive narrative."

    total_cust = len(df)
    top_persona = max(personas, key=lambda p: p["revenue_share"])
    largest_persona = max(personas, key=lambda p: p["count"])

    summary = f"""
    The customer base comprises <b>{total_cust:,} total active profiles</b> analyzed across <b>{len(personas)} distinct behavioral segments</b>.<br><br>
    • <b>Key Revenue Driver:</b> <span style="color: {top_persona['theme_color']}; font-weight: 700;">{top_persona['creative_name']}</span> accounts for <b>{top_persona['revenue_share']}%</b> of total company revenue despite representing only {top_persona['percentage']}% of the customer count.<br>
    • <b>Largest Volume Base:</b> <span style="color: {largest_persona['theme_color']}; font-weight: 700;">{largest_persona['creative_name']}</span> forms the largest group with <b>{largest_persona['count']:,} customers ({largest_persona['percentage']}%)</b>, making them prime candidates for cross-selling.<br>
    • <b>Strategic Action Item:</b> Prioritize retaining high-value segments with VIP perks while launching targeted promotional incentives for price-sensitive groups.
    """

    return summary
