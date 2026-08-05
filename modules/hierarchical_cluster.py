"""
Module: hierarchical_cluster.py
Description: Agglomerative Hierarchical Clustering Engine with Scipy Linkage Matrix calculation for Dendrograms.
"""

import numpy as np
from typing import Dict, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage


def run_hierarchical(scaled_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Runs Agglomerative Hierarchical Clustering and calculates linkage matrix for dendrogram visualization.
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(scaled_matrix)

    # Compute linkage matrix for dendrogram visualization
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
