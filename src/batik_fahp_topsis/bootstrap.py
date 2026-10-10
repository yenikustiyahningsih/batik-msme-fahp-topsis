"""Resampling bootstrap dan analisis stabilitas partisi (ARI / NMI), manuskrip Bagian 3.5."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .clustering import fit_kmeans
from .reproducibility import bootstrap_seed


def bootstrap_stability(
    X: np.ndarray,
    k: int,
    reference_labels: np.ndarray,
    n_bootstrap: int = 30,
    n_init: int = 10,
) -> dict:
    """Melakukan fit K-Means pada `n_bootstrap` resample (sampling dengan pengembalian, ukuran
    sama n) lalu membandingkan label hasil induksi setiap resample pada titik-titik full-data
    terhadap `reference_labels` melalui ARI dan NMI. Mengembalikan skor per repetisi serta
    ringkasan mean +/- std.
    """
    n = X.shape[0]
    aris, nmis = [], []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(bootstrap_seed(b))
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        km, _ = fit_kmeans(X_boot, k, seed=bootstrap_seed(b), n_init=n_init)
        # Memberi label pada titik-titik data penuh (asli) menggunakan centroid hasil fit
        # bootstrap, sehingga ARI/NMI dapat dihitung terhadap partisi acuan full-data yang tetap.
        boot_labels_on_full = km.predict(X)
        aris.append(adjusted_rand_score(reference_labels, boot_labels_on_full))
        nmis.append(normalized_mutual_info_score(reference_labels, boot_labels_on_full))

    aris = np.array(aris)
    nmis = np.array(nmis)
    return {
        "ARI_mean": float(aris.mean()),
        "ARI_std": float(aris.std(ddof=1)),
        "NMI_mean": float(nmis.mean()),
        "NMI_std": float(nmis.std(ddof=1)),
        "ARI_per_repetition": aris.tolist(),
        "NMI_per_repetition": nmis.tolist(),
    }


def cluster_size_distribution(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}
