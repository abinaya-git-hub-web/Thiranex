"""
Module: kmeans_cluster.py
Description: K-Means Machine Learning Clustering Engine with Elbow Method and Silhouette Evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(scaled_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Runs K-Means Clustering on normalized feature matrix.
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
    Computes Inertia (WCSS) and Silhouette Scores across a range of cluster values for K selection.
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
