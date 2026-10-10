"""Metode clustering kompetitor/baseline (manuskrip Bagian 3.5, Tabel 7, Tabel 9).

Mengimplementasikan: CLARA, FCM, IT2FCM, DBSCAN (via scikit-learn), dan weighted K-Means
(pembobotan jarak langsung dari FAHP / FAHP Type-2). Setiap hyperparameter dibaca dari
`params/baseline_params.yaml` (Reviewer 2, Komentar 3), bukan di-hard-code di sini.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

PARAMS_PATH = Path(__file__).resolve().parents[2] / "params" / "baseline_params.yaml"


def load_baseline_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------- #
# Weighted K-Means (pembobotan jarak langsung dari FAHP / FAHP Type-2)
# --------------------------------------------------------------------------------------- #
def weighted_kmeans(X: np.ndarray, weights: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """K-Means di mana jarak Euclidean kuadrat diskalakan per-fitur oleh `weights` (diterapkan
    sebagai pengali sqrt(weight) pada setiap kolom sebelum K-Means standar, setara dengan metrik
    Euclidean berbobot). `weights` sebaiknya berupa bobot kriteria FAHP atau FAHP Type-2 untuk
    kolom fitur yang dipertahankan pada `X`, dengan urutan kolom yang sama.
    """
    cfg = params["weighted_kmeans"]
    X_weighted = X * np.sqrt(weights)[None, :]
    km = KMeans(
        n_clusters=k,
        init=cfg["init"],
        n_init=cfg["n_init"],
        max_iter=cfg["max_iter"],
        random_state=cfg["random_state"],
    )
    labels = km.fit_predict(X_weighted)
    return labels, km.cluster_centers_


# --------------------------------------------------------------------------------------- #
# CLARA (Clustering LARge Applications)
# --------------------------------------------------------------------------------------- #
def _pam_cost(X: np.ndarray, medoid_indices: np.ndarray) -> float:
    d = pairwise_distances(X, X[medoid_indices])
    return float(d.min(axis=1).sum())


def _pam(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100) -> np.ndarray:
    """Partitioning Around Medoids versi minimal, dipakai sebagai solver internal CLARA pada
    setiap sub-sample."""
    n = X.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        d = pairwise_distances(X, X[medoids])
        labels = d.argmin(axis=1)
        improved = False
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                continue
            sub_d = pairwise_distances(X[members], X[members])
            costs = sub_d.sum(axis=1)
            best_local = members[np.argmin(costs)]
            if best_local != medoids[ci]:
                medoids[ci] = best_local
                improved = True
        if not improved:
            break
    return medoids


def clara(X: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """CLARA: berulang kali menarik sampel, menjalankan PAM pada tiap sampel, dan menyimpan
    himpunan medoid dengan biaya terendah yang dievaluasi terhadap keseluruhan dataset. Ukuran
    sampel mengikuti default standar PAM/CLARA: 40 + 2*k.
    """
    cfg = params["clara"]
    n = X.shape[0]
    sample_size = min(n, 40 + 2 * k)
    rng = np.random.default_rng(cfg["random_state"])

    best_medoid_points, best_cost = None, np.inf
    for s in range(cfg["n_samples"]):
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sub_rng = np.random.default_rng(cfg["random_state"] + s)
        local_medoid_idx = _pam(X[sample_idx], k, sub_rng)
        medoid_points = X[sample_idx][local_medoid_idx]

        d_full = pairwise_distances(X, medoid_points)
        cost = float(d_full.min(axis=1).sum())
        if cost < best_cost:
            best_cost, best_medoid_points = cost, medoid_points

    d_full = pairwise_distances(X, best_medoid_points)
    labels = d_full.argmin(axis=1)
    return labels, best_medoid_points


# --------------------------------------------------------------------------------------- #
# Fuzzy C-Means (via scikit-fuzzy)
# --------------------------------------------------------------------------------------- #
def fcm(X: np.ndarray, k: int, params: dict) -> dict:
    """Fuzzy C-Means. Mengembalikan label hard (argmax membership), matriks membership
    (n_samples x k), dan pusat cluster."""
    import skfuzzy as fuzz

    cfg = params["fcm"]
    np.random.seed(cfg["random_state"])
    centers, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T,
        c=k,
        m=cfg["fuzzy_exponent_m"],
        error=cfg["convergence_threshold"],
        maxiter=cfg["max_iter"],
        seed=cfg["random_state"],
    )
    labels = np.argmax(membership, axis=0)
    return {"labels": labels, "membership": membership.T, "centers": centers}


# --------------------------------------------------------------------------------------- #
# Interval Type-2 Fuzzy C-Means (custom; merata-ratakan dua hasil FCM pada m_lower / m_upper)
# --------------------------------------------------------------------------------------- #
def it2fcm(X: np.ndarray, k: int, params: dict) -> dict:
    """Interval Type-2 FCM: menjalankan FCM pada fuzzy exponent bawah dan atas (m_lower,
    m_upper) untuk memperoleh Lower/Upper Membership Function, lalu mereduksi tipe dengan
    merata-ratakan kedua matriks membership (penyederhanaan gaya centroid Karnik-Mendel)
    sebelum mengambil partisi hard sebagai argmax hasil reduksi tipe tersebut.
    """
    import skfuzzy as fuzz

    cfg = params["it2fcm"]
    np.random.seed(cfg["random_state"])

    centers_l, u_l, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, c=k, m=cfg["fuzzy_exponent_m_lower"], error=cfg["convergence_threshold"],
        maxiter=cfg["max_iter"], seed=cfg["random_state"],
    )
    centers_u, u_u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, c=k, m=cfg["fuzzy_exponent_m_upper"], error=cfg["convergence_threshold"],
        maxiter=cfg["max_iter"], seed=cfg["random_state"] + 1,
    )

    membership_reduced = (u_l + u_u) / 2.0
    labels = np.argmax(membership_reduced, axis=0)
    centers = (centers_l + centers_u) / 2.0
    return {
        "labels": labels,
        "membership": membership_reduced.T,
        "centers": centers,
        "umf": u_u.T,
        "lmf": u_l.T,
    }


# --------------------------------------------------------------------------------------- #
# DBSCAN
# --------------------------------------------------------------------------------------- #
def run_dbscan(X: np.ndarray, params: dict) -> np.ndarray:
    cfg = params["dbscan"]
    model = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"], metric=cfg["metric"])
    return model.fit_predict(X)


def k_distance_elbow(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Jarak k-nearest-neighbor yang sudah diurutkan, untuk memilih `eps` DBSCAN secara visual
    lewat metode elbow (manuskrip Bagian 3.10: "dipilih melalui pemeriksaan elbow k-distance")."""
    d = pairwise_distances(X)
    d.sort(axis=1)
    kth = d[:, min_samples]
    return np.sort(kth)
