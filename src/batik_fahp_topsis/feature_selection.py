"""Mekanisme pemilihan fitur yang dibandingkan pada studi ablasi (Tabel 9, Bagian 3.10).

- `nested_fahp_subsets`: N kriteria berperingkat teratas hasil FAHP, untuk N = 7..16 (mekanisme
  yang diusulkan).
- `random_subsets`: 100 subset acak berisi 7 fitur, diambil dari 15 indikator nonkonstan
  (C10 konstan di seluruh dataset sehingga dikeluarkan, manuskrip Bagian 3.2).
- `sequential_backward_selection`: baseline SBS yang murni data-driven, dipakai untuk menguji
  apakah subset hasil bimbingan pakar juga merupakan subset yang optimal secara statistik.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .reproducibility import MAIN_GRID_SEED, get_rng


def nested_fahp_subsets(ranked_criteria: list[str], n_values: range = range(7, 17)) -> dict[int, list[str]]:
    """N kriteria teratas dari hasil ranking FAHP, untuk setiap N pada `n_values`."""
    return {n: ranked_criteria[:n] for n in n_values}


def random_subsets(
    candidate_criteria: list[str],
    subset_size: int = 7,
    n_subsets: int = 100,
    seed: int = MAIN_GRID_SEED,
) -> list[list[str]]:
    """`n_subsets` subset acak berukuran `subset_size`, diambil tanpa pengembalian per subset
    dari `candidate_criteria` (15 indikator nonkonstan, yaitu seluruh 16 indikator dikurangi C10)."""
    rng = get_rng(seed)
    subsets = []
    for _ in range(n_subsets):
        subsets.append(sorted(rng.choice(candidate_criteria, size=subset_size, replace=False).tolist()))
    return subsets


def sequential_backward_selection(
    X_full: np.ndarray,
    feature_names: list[str],
    k: int,
    floor_n_features: int = 7,
    seed: int = MAIN_GRID_SEED,
) -> list[str]:
    """Sequential Backward Selection (SBS) yang greedy, dinilai dengan Silhouette Coefficient
    dari hasil fit K-Means(k), berhenti setelah tersisa `floor_n_features` fitur (manuskrip:
    "greedy sequential backward selection").

    Pada setiap langkah, fitur yang penghapusannya paling meningkatkan (atau paling sedikit
    menurunkan) SC akan dibuang.
    """
    remaining = list(range(len(feature_names)))

    def score(indices: list[int]) -> float:
        if len(indices) < 2:
            return -np.inf
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300, random_state=seed)
        labels = km.fit_predict(X_full[:, indices])
        if len(set(labels)) < 2:
            return -np.inf
        return silhouette_score(X_full[:, indices], labels)

    while len(remaining) > floor_n_features:
        best_candidate, best_score = None, -np.inf
        for drop in remaining:
            trial = [i for i in remaining if i != drop]
            s = score(trial)
            if s > best_score:
                best_score, best_candidate = s, drop
        remaining = [i for i in remaining if i != best_candidate]

    return [feature_names[i] for i in remaining]
