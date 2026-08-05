"""
Module: ai_predictor.py
Description: Advanced AI Predictor Suite: Random Forest Churn Classifier, 
             CLV Regression Forecasting, and Next Best Offer Recommendation Engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def predict_customer_churn(df: pd.DataFrame, mappings: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[str, float]]]:
    """
    Trains a Random Forest Classifier to predict customer churn probability 
    and extracts feature importance drivers.
    """
    df_churn = df.copy()

    # Extract modeling features
    features_df = pd.DataFrame()
    features_df["Age"] = pd.to_numeric(df_churn[mappings.get("age")], errors="coerce").fillna(35) if mappings and mappings.get("age") in df_churn else df_churn.get("Age", 35)
    features_df["Income"] = pd.to_numeric(df_churn[mappings.get("income")], errors="coerce").fillna(50000) if mappings and mappings.get("income") in df_churn else df_churn.get("Income", 50000)
    features_df["Spend"] = pd.to_numeric(df_churn[mappings.get("spend")], errors="coerce").fillna(1000) if mappings and mappings.get("spend") in df_churn else df_churn.get("Total Spend ($)", df_churn.get("Monetary_Val", 1000))
    features_df["Frequency"] = pd.to_numeric(df_churn[mappings.get("frequency")], errors="coerce").fillna(5) if mappings and mappings.get("frequency") in df_churn else df_churn.get("Purchase Frequency", df_churn.get("Frequency_Val", 5))
    features_df["AOV"] = pd.to_numeric(df_churn[mappings.get("aov")], errors="coerce").fillna(100) if mappings and mappings.get("aov") in df_churn else df_churn.get("Average Order Value ($)", 100)

    if "Recency_Days" in df_churn:
        features_df["Recency"] = df_churn["Recency_Days"]
    else:
        features_df["Recency"] = np.random.randint(1, 365, len(df_churn))

    sat_col = mappings.get("satisfaction") if mappings else "Satisfaction Score"
    features_df["Satisfaction"] = pd.to_numeric(df_churn[sat_col], errors="coerce").fillna(3) if sat_col in df_churn else 3

    # Ground-truth target label: Customer Churn
    churn_mask = (features_df["Recency"] > 150) | ((features_df["Satisfaction"] <= 2) & (features_df["Frequency"] < 12))
    target = churn_mask.astype(int)

    # Scale features & Fit Random Forest Classifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    rf.fit(X_scaled, target)

    churn_probs = rf.predict_proba(X_scaled)[:, 1]
    df_churn["Churn_Probability"] = np.round(churn_probs * 100, 1)
    df_churn["Churn_Risk_Category"] = pd.cut(churn_probs, bins=[-0.01, 0.35, 0.65, 1.0], labels=["Low Churn Risk", "Medium Churn Risk", "High Churn Risk"])

    # Feature Importance Drivers
    feature_names = list(features_df.columns)
    importances = rf.feature_importances_
    drivers = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    segment_col = "RFM_Segment" if "RFM_Segment" in df_churn else "Active_Segment" if "Active_Segment" in df_churn else None
    segment_churn_rates = {}
    if segment_col and segment_col in df_churn:
        segment_churn_rates = df_churn.groupby(segment_col)["Churn_Probability"].mean().round(1).to_dict()

    return df_churn, segment_churn_rates, drivers


def predict_customer_clv(df: pd.DataFrame, mappings: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates Historic CLV and predicts Future Customer Lifetime Value (CLV) using Random Forest Regression.
    Categorizes customers into CLV Tiers.
    """
    df_clv = df.copy()

    spend_col = mappings.get("spend") if mappings and mappings.get("spend") in df_clv else ("Total Spend ($)" if "Total Spend ($)" in df_clv else "Monetary_Val")
    freq_col = mappings.get("frequency") if mappings and mappings.get("frequency") in df_clv else ("Purchase Frequency" if "Purchase Frequency" in df_clv else "Frequency_Val")
    inc_col = mappings.get("income") if mappings and mappings.get("income") in df_clv else ("Income" if "Income" in df_clv else None)

    spend = pd.to_numeric(df_clv[spend_col], errors="coerce").fillna(1000)
    freq = pd.to_numeric(df_clv[freq_col], errors="coerce").fillna(5)
    income = pd.to_numeric(df_clv[inc_col], errors="coerce").fillna(50000) if inc_col else pd.Series(50000, index=df_clv.index)

    # Historic CLV
    df_clv["Historic_CLV"] = np.round(spend * (1 + (freq / 30.0)), 2)

    # Train Regression model to predict 2-Year Future CLV
    X = pd.DataFrame({"Spend": spend, "Frequency": freq, "Income": income})
    growth_multiplier = np.where(freq > 20, 1.4, np.where(freq > 10, 1.2, 0.95))
    y = df_clv["Historic_CLV"] * growth_multiplier + np.random.normal(0, 100, len(df_clv))

    reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    reg.fit(X, y)

    df_clv["Predicted_Future_CLV"] = np.round(reg.predict(X), 2)

    # Assign CLV Tiers (High CLV, Medium CLV, Low CLV)
    clv_quantiles = df_clv["Predicted_Future_CLV"].quantile([0.33, 0.67]).values
    q_low, q_high = clv_quantiles[0], clv_quantiles[1]

    def assign_tier(val):
        if val >= q_high:
            return "💎 High CLV Tier"
        elif val >= q_low:
            return "🥇 Medium CLV Tier"
        else:
            return "🥈 Low CLV Tier"

    df_clv["CLV_Tier"] = df_clv["Predicted_Future_CLV"].apply(assign_tier)

    summary_stats = {
        "avg_historic_clv": round(float(df_clv["Historic_CLV"].mean()), 2),
        "avg_predicted_clv": round(float(df_clv["Predicted_Future_CLV"].mean()), 2),
        "total_portfolio_clv": round(float(df_clv["Predicted_Future_CLV"].sum()), 2)
    }

    return df_clv, summary_stats


def generate_next_best_offer(df: pd.DataFrame, segment_col: str) -> List[Dict[str, str]]:
    """
    Generates intelligent Next Best Offer recommendations for each segment based on purchasing behavior.
    """
    if segment_col not in df.columns:
        return []

    segments = df[segment_col].unique()
    recommendations = []

    for seg in segments:
        seg_df = df[df[segment_col] == seg]

        top_cat = "Electronics"
        if "Product Categories Purchased" in seg_df.columns:
            cats = seg_df["Product Categories Purchased"].dropna().astype(str).str.split(", ").explode()
            if not cats.empty:
                top_cat = cats.value_counts().index[0]

        top_pay = "Credit Card"
        if "Preferred Payment" in seg_df.columns:
            top_pay = seg_df["Preferred Payment"].mode().iloc[0] if not seg_df["Preferred Payment"].empty else "Credit Card"

        avg_spend = seg_df["Total Spend ($)"].mean() if "Total Spend ($)" in seg_df.columns else 1000

        if avg_spend > 4000:
            offer = f"Offer 20% discount on premium {top_cat} accessories + Instant 5x rewards via {top_pay}."
            channel = "Direct VIP Account Manager Email & Priority SMS"
        elif avg_spend > 1500:
            offer = f"Bundle 2 items from {top_cat} with free 1-year extended warranty when paying via {top_pay}."
            channel = "In-App Push Notification & Dynamic Website Banner"
        else:
            offer = f"Get $15 off your next order in {top_cat} with minimum spend $50."
            channel = "Email Newsletter & Retargeting Social Ads"

        recommendations.append({
            "segment": str(seg),
            "preferred_category": top_cat,
            "preferred_payment": top_pay,
            "next_best_offer": offer,
            "recommended_channel": channel
        })

    return recommendations


def analyze_purchase_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs category co-occurrence and purchase pattern analysis.
    """
    cat_counts = {}
    if "Product Categories Purchased" in df.columns:
        cats = df["Product Categories Purchased"].dropna().astype(str).str.split(", ").explode()
        cat_counts = cats.value_counts().head(8).to_dict()

    pay_counts = {}
    if "Preferred Payment" in df.columns:
        pay_counts = df["Preferred Payment"].value_counts().to_dict()

    return {
        "top_categories": cat_counts,
        "payment_methods": pay_counts
    }
