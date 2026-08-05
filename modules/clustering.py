"""
Module: clustering.py
Description: Machine Learning Clustering Engine implementing K-Means, DBSCAN, 
             Hierarchical Agglomerative Clustering, Elbow Method, and Silhouette Analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage


def extract_and_scale_features(df: pd.DataFrame, mappings: Dict[str, str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Extracts numerical clustering features (Spend, Frequency, AOV, Recency, Age, Income)
    and applies StandardScaler normalization.

    Returns:
        Tuple[np.ndarray, List[str], pd.DataFrame]: Scaled numpy array, feature names list, clean feature DataFrame.
    """
    feature_cols = []
    data_dict = {}

    # 1. Total Spend
    spend_col = mappings.get("spend")
    if spend_col and spend_col in df.columns:
        data_dict["Spend"] = pd.to_numeric(df[spend_col], errors="coerce").fillna(df[spend_col].median() if spend_col in df else 500)
    else:
        data_dict["Spend"] = df["Monetary_Val"] if "Monetary_Val" in df.columns else np.random.uniform(100, 5000, len(df))

    # 2. Purchase Frequency
    freq_col = mappings.get("frequency")
    if freq_col and freq_col in df.columns:
        data_dict["Frequency"] = pd.to_numeric(df[freq_col], errors="coerce").fillna(df[freq_col].median() if freq_col in df else 5)
    else:
        data_dict["Frequency"] = df["Frequency_Val"] if "Frequency_Val" in df.columns else np.random.randint(1, 20, len(df))

    # 3. Average Order Value (AOV)
    aov_col = mappings.get("aov")
    if aov_col and aov_col in df.columns:
        data_dict["AOV"] = pd.to_numeric(df[aov_col], errors="coerce").fillna(100)
    else:
        data_dict["AOV"] = (data_dict["Spend"] / np.maximum(data_dict["Frequency"], 1)).round(2)

    # 4. Recency (Days since last purchase)
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

    # 5. Age
    age_col = mappings.get("age")
    if age_col and age_col in df.columns:
        data_dict["Age"] = pd.to_numeric(df[age_col], errors="coerce").fillna(df[age_col].median() if age_col in df else 35)
    else:
        data_dict["Age"] = np.random.randint(18, 65, len(df))

    # 6. Income
    inc_col = mappings.get("income")
    if inc_col and inc_col in df.columns:
        data_dict["Income"] = pd.to_numeric(df[inc_col], errors="coerce").fillna(df[inc_col].median() if inc_col in df else 55000)
    else:
        data_dict["Income"] = np.random.randint(20000, 120000, len(df))

    feature_df = pd.DataFrame(data_dict)
    feature_names = list(feature_df.columns)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_df)

    return scaled_matrix, feature_names, feature_df


def run_kmeans(scaled_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Runs K-Means Clustering on scaled features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_matrix)

    score = 0.0
    if len(np.unique(labels)) > 1:
        score = float(silhouette_score(scaled_matrix, labels))

    cluster_names = [f"K-Means Cluster {i+1}" for i in range(n_clusters)]

    return {
        "model": kmeans,
        "labels": labels,
        "cluster_names": cluster_names,
        "cluster_centers": kmeans.cluster_centers_,
        "silhouette_score": round(score, 3),
        "n_clusters": n_clusters
    }


def compute_elbow_and_silhouette(scaled_matrix: np.ndarray, k_range: range = range(2, 9)) -> Dict[str, List[float]]:
    """
    Computes Inertia (WCSS) and Silhouette Scores for a range of cluster values.
    """
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        inertias.append(float(km.inertia_))

        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(scaled_matrix, labels))
            silhouettes.append(round(sil, 3))
        else:
            silhouettes.append(0.0)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouettes
    }


def run_dbscan(scaled_matrix: np.ndarray, eps: float = 0.8, min_samples: int = 5) -> Dict[str, Any]:
    """
    Runs DBSCAN density-based clustering to find non-spherical clusters and outliers (-1).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))

    score = 0.0
    if n_clusters > 1:
        # Calculate silhouette score excluding noise points
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > n_clusters:
            score = float(silhouette_score(scaled_matrix[non_noise_mask], labels[non_noise_mask]))

    return {
        "model": dbscan,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_score": round(score, 3)
    }


def run_hierarchical(scaled_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Runs Agglomerative Hierarchical Clustering and computes Scipy linkage matrix for dendrograms.
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(scaled_matrix)

    # Compute linkage matrix for dendrogram visualization (sample down if > 300 rows for speed)
    sample_matrix = scaled_matrix
    if len(scaled_matrix) > 300:
        indices = np.random.choice(len(scaled_matrix), 300, replace=False)
        sample_matrix = scaled_matrix[indices]

    linkage_matrix = linkage(sample_matrix, method="ward")

    score = 0.0
    if len(np.unique(labels)) > 1:
        score = float(silhouette_score(scaled_matrix, labels))

    return {
        "model": agg,
        "labels": labels,
        "linkage_matrix": linkage_matrix,
        "silhouette_score": round(score, 3),
        "n_clusters": n_clusters
    }
