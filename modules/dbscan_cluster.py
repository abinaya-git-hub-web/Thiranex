"""
Module: dbscan_cluster.py
Description: DBSCAN Density-Based Clustering Engine to detect non-spherical clusters and outliers/spikes.
"""

import numpy as np
from typing import Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run_dbscan(scaled_matrix: np.ndarray, eps: float = 0.8, min_samples: int = 5) -> Dict[str, Any]:
    """
    Runs DBSCAN density-based clustering to find clusters and detect noise (-1).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))

    score = 0.0
    if n_clusters > 1:
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
