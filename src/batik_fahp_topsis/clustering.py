"""Grid search gabungan fitur-cluster dengan K-Means (manuskrip Bagian 3.4, Algoritma 1).

`run_joint_grid` mengevaluasi setiap kombinasi (N, K) -- N dalam {7, ..., 16} subset fitur
bertingkat hasil ranking FAHP, K dalam {2, ..., 10} -- melakukan fit K-Means pada representasi
fitur yang sama (sudah dinormalisasi Min-Max), dan mencatat SC / DBI / CHI untuk setiap 90
konfigurasi. Ini adalah sumber otoritatif tunggal untuk Tabel 5 / Tabel 6 / Figure 2 / Figure 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .reproducibility import MAIN_GRID_SEED
from .validity_metrics import internal_validity


@dataclass
class ConfigResult:
    n_features: int
    k: int
    features: list[str]
    sc: float
    dbi: float
    chi: float
    labels: np.ndarray = field(repr=False)


def fit_kmeans(X: np.ndarray, k: int, seed: int = MAIN_GRID_SEED, n_init: int = 20, max_iter: int = 300):
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def run_joint_grid(
    df_scaled: pd.DataFrame,
    nested_subsets: dict[int, list[str]],
    k_values: range = range(2, 11),
    seed: int = MAIN_GRID_SEED,
) -> list[ConfigResult]:
    """Mengevaluasi setiap konfigurasi (N, K) pada grid gabungan.

    `nested_subsets` memetakan N -> daftar N nama fitur hasil ranking FAHP (lihat
    `feature_selection.nested_fahp_subsets`).
    """
    results: list[ConfigResult] = []
    for n_features, feature_names in nested_subsets.items():
        X = df_scaled[feature_names].to_numpy()
        for k in k_values:
            _, labels = fit_kmeans(X, k, seed=seed)
            metrics = internal_validity(X, labels)
            results.append(
                ConfigResult(
                    n_features=n_features,
                    k=k,
                    features=feature_names,
                    sc=metrics["SC"],
                    dbi=metrics["DBI"],
                    chi=metrics["CHI"],
                    labels=labels,
                )
            )
    return results


def results_to_dataframe(results: list[ConfigResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_features": r.n_features,
                "k": r.k,
                "features": ",".join(r.features),
                "SC": r.sc,
                "DBI": r.dbi,
                "CHI": r.chi,
            }
            for r in results
        ]
    )
