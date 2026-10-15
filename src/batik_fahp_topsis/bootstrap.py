
"""Resampling bootstrap dan analisis stabilitas partisi (ARI / NMI),
manuskrip Bagian 3.5.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .clustering import fit_kmeans
from .reproducibility import bootstrap_seed


def bootstrap_stability(
    X: np.ndarray,
    k: int,
    reference_labels: np.ndarray,
    n_bootstrap: int = 30,
    n_init: int = 10,
) -> dict:
    """Mengukur stabilitas partisi melalui resampling bootstrap.

    Setiap repetisi mengambil sampel berukuran n dengan pengembalian,
    melatih K-Means pada sampel tersebut, lalu memprediksi label seluruh
    data asli menggunakan model hasil bootstrap.

    Label prediksi dibandingkan dengan reference_labels menggunakan ARI
    dan NMI. Fungsi mengembalikan mean, sample standard deviation, serta
    skor pada setiap repetisi.
    """
    X = np.asarray(X, dtype=float)
    reference_labels = np.asarray(reference_labels)

    # Validasi data dan parameter.
    if X.ndim != 2:
        raise ValueError("X harus berupa matriks dua dimensi.")

    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError("X harus memiliki minimal dua sampel.")
    if n_features < 1:
        raise ValueError("X harus memiliki minimal satu fitur.")
    if not np.isfinite(X).all():
        raise ValueError("X mengandung NaN atau nilai tak hingga.")

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError("k harus berupa bilangan bulat.")
    if not 2 <= k <= n_samples:
        raise ValueError("k harus berada di antara 2 dan jumlah sampel.")

    if reference_labels.ndim != 1:
        raise ValueError("reference_labels harus berupa array satu dimensi.")
    if len(reference_labels) != n_samples:
        raise ValueError(
            "Panjang reference_labels harus sama dengan jumlah sampel X."
        )
    if pd_is_missing(reference_labels):
        raise ValueError("reference_labels mengandung label kosong.")

    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(
        n_bootstrap, bool
    ):
        raise ValueError("n_bootstrap harus berupa bilangan bulat.")
    if n_bootstrap < 2:
        raise ValueError(
            "n_bootstrap minimal 2 agar standar deviasi sampel dapat dihitung."
        )

    if not isinstance(n_init, (int, np.integer)) or isinstance(n_init, bool):
        raise ValueError("n_init harus berupa bilangan bulat.")
    if n_init < 1:
        raise ValueError("n_init harus minimal 1.")

    aris = []
    nmis = []

    for b in range(n_bootstrap):
        seed = bootstrap_seed(b)
        rng = np.random.default_rng(seed)

        # Bootstrap sample: sampling dengan pengembalian.
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot = X[idx]

        # Fit K-Means pada sampel bootstrap.
        km, _ = fit_kmeans(
            X_boot,
            k,
            seed=seed,
            n_init=n_init,
        )

        # Prediksi seluruh data asli agar dapat dibandingkan dengan
        # partisi referensi yang memiliki panjang sama.
        boot_labels_on_full = km.predict(X)

        ari = adjusted_rand_score(
            reference_labels,
            boot_labels_on_full,
        )
        nmi = normalized_mutual_info_score(
            reference_labels,
            boot_labels_on_full,
        )

        aris.append(float(ari))
        nmis.append(float(nmi))

    aris = np.asarray(aris, dtype=float)
    nmis = np.asarray(nmis, dtype=float)

    return {
        "ARI_mean": float(aris.mean()),
        "ARI_std": float(aris.std(ddof=1)),
        "NMI_mean": float(nmis.mean()),
        "NMI_std": float(nmis.std(ddof=1)),
        "ARI_per_repetition": aris.tolist(),
        "NMI_per_repetition": nmis.tolist(),
    }


def pd_is_missing(values: np.ndarray) -> bool:
    """Memeriksa label kosong tanpa bergantung pada pandas."""
    try:
        missing = np.asarray(
            [
                value is None
                or (
                    isinstance(value, (float, np.floating))
                    and np.isnan(value)
                )
                for value in values
            ],
            dtype=bool,
        )
        return bool(missing.any())
    except (TypeError, ValueError):
        return False


def cluster_size_distribution(labels: np.ndarray) -> dict[int, int]:
    """Menghitung jumlah anggota setiap cluster."""
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels harus berupa array satu dimensi.")
    if len(labels) == 0:
        raise ValueError("labels tidak boleh kosong.")
    if pd_is_missing(labels):
        raise ValueError("labels mengandung label kosong.")

    unique, counts = np.unique(labels, return_counts=True)

    return {
        int(cluster_id): int(count)
        for cluster_id, count in zip(unique, counts)
    }
