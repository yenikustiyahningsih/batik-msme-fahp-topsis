"""Metrik validitas cluster internal dan validitas khusus-fuzzy.

SC (Silhouette Coefficient, makin tinggi makin baik), DBI (Davies-Bouldin Index, makin
rendah makin baik), dan CHI (Calinski-Harabasz Index, makin tinggi makin baik) adalah tiga
indeks internal utama yang dipakai di seluruh optimasi gabungan dan perbandingan baseline.
PC (Partition Coefficient), PE (Partition Entropy), dan XB (Xie-Beni index) adalah ukuran
validitas khusus-fuzzy yang dilaporkan untuk FCM/IT2FCM di Tabel 10, dilengkapi dengan FOU
(lihat `fahp.footprint_of_uncertainty`).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def internal_validity(X: np.ndarray, labels: np.ndarray) -> dict:
    """SC, DBI, CHI untuk sebuah partisi hard yang sudah di-fit. Mengembalikan NaN jika partisi
    degenerate (kurang dari 2 cluster non-noise, seperti yang bisa terjadi pada DBSCAN)."""
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return {"SC": float("nan"), "DBI": float("nan"), "CHI": float("nan")}
    mask = labels != -1
    return {
        "SC": float(silhouette_score(X[mask], labels[mask])),
        "DBI": float(davies_bouldin_score(X[mask], labels[mask])),
        "CHI": float(calinski_harabasz_score(X[mask], labels[mask])),
    }


def partition_coefficient(membership: np.ndarray) -> float:
    """PC = (1/n) * sum_i sum_c u_ic^2. Bentuk membership: (n_samples, n_clusters)."""
    n = membership.shape[0]
    return float(np.sum(membership ** 2) / n)


def partition_entropy(membership: np.ndarray, eps: float = 1e-12) -> float:
    """PE = -(1/n) * sum_i sum_c u_ic * log(u_ic)."""
    n = membership.shape[0]
    safe = np.clip(membership, eps, 1.0)
    return float(-np.sum(membership * np.log(safe)) / n)


def xie_beni_index(X: np.ndarray, membership: np.ndarray, centers: np.ndarray, m: float = 2.0) -> float:
    """XB = sum_i sum_c u_ic^m * ||x_i - v_c||^2 / (n * min_{c != c'} ||v_c - v_c'||^2)."""
    n = X.shape[0]
    dist_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (n, c)
    numerator = np.sum((membership ** m) * dist_sq)

    c = centers.shape[0]
    min_center_dist_sq = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            d = float(np.sum((centers[i] - centers[j]) ** 2))
            if d < min_center_dist_sq:
                min_center_dist_sq = d
    denominator = n * min_center_dist_sq
    return float(numerator / denominator) if denominator > 0 else float("nan")
