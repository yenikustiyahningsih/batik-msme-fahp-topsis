
"""TOPSIS khusus per-cluster dan analisis sensitivitas (manuskrip Bagian 3.6, Tabel 12, Tabel 13).

Skor TOPSIS dihitung di dalam masing-masing cluster akhir secara terpisah,
bukan secara global. Skor dari satu cluster tidak boleh dibandingkan langsung
dengan skor dari cluster lain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def topsis_within_cluster(
    X: np.ndarray,
    weights: np.ndarray,
    cost_mask: np.ndarray,
) -> np.ndarray:
    """Menghitung skor TOPSIS untuk alternatif dalam satu cluster.

    X: matriks (n_alternatif, n_kriteria).
    weights: bobot kriteria yang dinormalisasi.
    cost_mask: True untuk kriteria cost dan False untuk benefit.

    Mengembalikan closeness coefficient TOPSIS dalam rentang [0, 1].
    Skor yang lebih tinggi menunjukkan alternatif yang lebih diprioritaskan.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cost_mask = np.asarray(cost_mask, dtype=bool)

    if X.ndim != 2:
        raise ValueError("X harus berupa matriks dua dimensi.")
    if X.shape[0] == 0:
        return np.array([], dtype=float)
    if X.shape[1] == 0:
        raise ValueError("X harus memiliki minimal satu kriteria.")
    if weights.shape != (X.shape[1],):
        raise ValueError("Jumlah bobot harus sama dengan jumlah kriteria.")
    if cost_mask.shape != (X.shape[1],):
        raise ValueError(
            "cost_mask harus sama panjang dengan jumlah kriteria."
        )
    if not np.isfinite(X).all():
        raise ValueError("X mengandung NaN atau nilai tak hingga.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Bobot harus finite dan nonnegatif.")
    if weights.sum() <= 0:
        raise ValueError("Jumlah bobot harus lebih besar dari nol.")

    weights = weights / weights.sum()

    # Normalisasi vektor dan pembobotan.
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    V = (X / norms) * weights[None, :]

    # Solusi ideal terbaik dan terburuk.
    ideal_best = np.where(
        cost_mask, V.min(axis=0), V.max(axis=0)
    )
    ideal_worst = np.where(
        cost_mask, V.max(axis=0), V.min(axis=0)
    )

    dist_best = np.sqrt(
        ((V - ideal_best[None, :]) ** 2).sum(axis=1)
    )
    dist_worst = np.sqrt(
        ((V - ideal_worst[None, :]) ** 2).sum(axis=1)
    )

    denom = dist_best + dist_worst

    # Jika solusi ideal berimpit, skor semua alternatif netral.
    scores = np.full(len(X), 0.5, dtype=float)
    valid = denom > 0
    scores[valid] = dist_worst[valid] / denom[valid]

    return scores


def cluster_specific_topsis(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    criteria: list[str],
    weights: np.ndarray,
    cost_criteria: list[str],
    id_column: str = "msme_id",
) -> pd.DataFrame:
    """Menghitung TOPSIS secara terpisah pada setiap cluster.

    Menghasilkan satu baris per MSME dengan informasi cluster,
    skor TOPSIS, dan peringkat di dalam cluster.
    Label -1 dilewati sebagai noise.
    """
    labels = np.asarray(labels)

    if len(df_scaled) != len(labels):
        raise ValueError(
            "Jumlah baris data harus sama dengan jumlah label cluster."
        )
    if id_column not in df_scaled.columns:
        raise ValueError(f"Kolom ID '{id_column}' tidak ditemukan.")

    missing = set(criteria).difference(df_scaled.columns)
    if missing:
        raise ValueError(
            f"Kolom kriteria tidak ditemukan: {sorted(missing)}"
        )
    if df_scaled[id_column].duplicated().any():
        raise ValueError(f"Kolom ID '{id_column}' mengandung duplikat.")

    cost_mask = np.array(
        [criterion in cost_criteria for criterion in criteria],
        dtype=bool,
    )
    out_rows = []

    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue

        member_mask = labels == cluster_id
        X = df_scaled.loc[member_mask, criteria].to_numpy(dtype=float)
        scores = topsis_within_cluster(X, weights, cost_mask)

        sub = pd.DataFrame(
            {
                id_column: df_scaled.loc[
                    member_mask, id_column
                ].to_numpy(),
                "cluster": cluster_id,
                "topsis_score": scores,
            }
        )

        # method="min": skor identik mendapatkan peringkat yang sama.
        sub["rank_within_cluster"] = (
            sub["topsis_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        out_rows.append(sub)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                id_column,
                "cluster",
                "topsis_score",
                "rank_within_cluster",
            ]
        )

    return (
        pd.concat(out_rows, ignore_index=True)
        .sort_values(
            ["cluster", "rank_within_cluster", id_column],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def perturb_weights(
    weights: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    signed: bool = False,
) -> np.ndarray:
    """Mengganggu bobot sebesar +/- fraction dan menormalisasi ulang.

    signed=False: gangguan uniform antara -fraction dan +fraction.
    signed=True: besar gangguan tepat fraction dengan tanda acak.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights harus berupa array satu dimensi nonkosong.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Bobot harus finite dan nonnegatif.")
    if weights.sum() <= 0:
        raise ValueError("Jumlah bobot harus lebih besar dari nol.")
    if not 0 <= fraction < 1:
        raise ValueError("fraction harus berada pada rentang [0, 1).")

    if signed:
        noise = fraction * rng.choice(
            [-1.0, 1.0], size=weights.shape
        )
    else:
        noise = rng.uniform(
            -fraction, fraction, size=weights.shape
        )

    perturbed = np.clip(weights * (1.0 + noise), 1e-6, None)
    return perturbed / perturbed.sum()


def equal_weight_stress_test(n_criteria: int) -> np.ndarray:
    """Menghasilkan bobot sama untuk uji tekanan."""
    if n_criteria < 1:
        raise ValueError("n_criteria harus minimal 1.")

    return np.full(n_criteria, 1.0 / n_criteria)


def _top_k_indices_with_ties(
    scores: np.ndarray,
    top_k: int,
) -> set[int]:
    """Memilih top-K dengan menyertakan seluruh skor seri pada batas K."""
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("scores harus berupa array satu dimensi.")
    if not np.isfinite(scores).all():
        raise ValueError("scores mengandung NaN atau nilai tak hingga.")
    if top_k < 1:
        raise ValueError("top_k harus minimal 1.")
    if len(scores) == 0:
        return set()

    k = min(int(top_k), len(scores))
    cutoff = np.sort(scores)[::-1][k - 1]

    return set(np.flatnonzero(scores >= cutoff).tolist())


def _per_cluster_stability(
    base_scores,
    new_scores,
    top_k,
):
    """Menghitung stabilitas ranking per cluster.

    Mengembalikan mean Spearman rho, minimum Spearman rho,
    dan rata-rata Jaccard similarity untuk himpunan top-K.
    Skor seri pada batas K disertakan seluruhnya.
    """
    from scipy.stats import spearmanr

    if len(base_scores) != len(new_scores):
        raise ValueError(
            "Jumlah cluster pada skor dasar dan skor baru berbeda."
        )
    if top_k < 1:
        raise ValueError("top_k harus minimal 1.")

    rhos = []
    overlaps = []

    for base, perturbed in zip(base_scores, new_scores):
        base = np.asarray(base, dtype=float)
        perturbed = np.asarray(perturbed, dtype=float)

        if base.ndim != 1 or perturbed.ndim != 1:
            raise ValueError("Skor setiap cluster harus satu dimensi.")
        if len(base) != len(perturbed):
            raise ValueError(
                "Jumlah alternatif pada cluster tidak konsisten."
            )
        if not np.isfinite(base).all() or not np.isfinite(
            perturbed
        ).all():
            raise ValueError("Skor mengandung NaN atau nilai tak hingga.")

        if len(base) < 2:
            rhos.append(np.nan)
        else:
            result = spearmanr(base, perturbed)
            rho = result.statistic
            rhos.append(
                float(rho) if np.isfinite(rho) else np.nan
            )

        top_base = _top_k_indices_with_ties(base, top_k)
        top_new = _top_k_indices_with_ties(perturbed, top_k)

        union = top_base | top_new
        jaccard = (
            len(top_base & top_new) / len(union)
            if union
            else 1.0
        )
        overlaps.append(jaccard)

    valid_rhos = np.asarray(rhos, dtype=float)
    valid_rhos = valid_rhos[np.isfinite(valid_rhos)]

    mean_rho = (
        float(valid_rhos.mean())
        if len(valid_rhos)
        else float("nan")
    )
    min_rho = (
        float(valid_rhos.min())
        if len(valid_rhos)
        else float("nan")
    )
    mean_overlap = (
        float(np.mean(overlaps))
        if overlaps
        else float("nan")
    )

    return mean_rho, min_rho, mean_overlap


def sensitivity_summary(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    criteria: list[str],
    base_weights: np.ndarray,
    cost_criteria: list[str],
    fractions: tuple[float, ...] = (0.05, 0.10, 0.15),
    n_repeats: int = 100,
    seed: int = 42,
    top_k: int = 3,
    signed: bool = True,
) -> pd.DataFrame:
    """Tabel sensitivitas TOPSIS berdasarkan gangguan bobot.

    mean_rho: rata-rata Spearman rho lintas cluster dan ulangan.
    min_rho: rho minimum lintas cluster dan ulangan.
    top3_overlap: rata-rata Jaccard similarity top-K per cluster.

    Catatan: top3_overlap menggunakan definisi Jaccard dan menyertakan
    semua alternatif yang seri pada batas top-K.
    """
    labels = np.asarray(labels)
    base_weights = np.asarray(base_weights, dtype=float)

    if len(df_scaled) != len(labels):
        raise ValueError(
            "Jumlah baris data harus sama dengan jumlah label cluster."
        )
    if n_repeats < 1:
        raise ValueError("n_repeats harus minimal 1.")
    if top_k < 1:
        raise ValueError("top_k harus minimal 1.")
    if len(criteria) == 0:
        raise ValueError("criteria harus berisi minimal satu kriteria.")
    if base_weights.shape != (len(criteria),):
        raise ValueError(
            "Jumlah bobot harus sama dengan jumlah kriteria."
        )
    if not np.isfinite(base_weights).all() or (
        base_weights < 0
    ).any():
        raise ValueError("Bobot dasar harus finite dan nonnegatif.")
    if base_weights.sum() <= 0:
        raise ValueError("Jumlah bobot dasar harus lebih besar dari nol.")

    base_weights = base_weights / base_weights.sum()
    cost_mask = np.array(
        [criterion in cost_criteria for criterion in criteria],
        dtype=bool,
    )

    missing = set(criteria).difference(df_scaled.columns)
    if missing:
        raise ValueError(
            f"Kolom kriteria tidak ditemukan: {sorted(missing)}"
        )

    clusters = [
        cluster_id
        for cluster_id in sorted(set(labels.tolist()))
        if cluster_id != -1
    ]
    Xs = [
        df_scaled.loc[labels == cluster_id, criteria].to_numpy(
            dtype=float
        )
        for cluster_id in clusters
    ]

    base = [
        topsis_within_cluster(X, base_weights, cost_mask)
        for X in Xs
    ]

    def evaluate(weights):
        new_scores = [
            topsis_within_cluster(X, weights, cost_mask)
            for X in Xs
        ]
        return _per_cluster_stability(
            base, new_scores, top_k
        )

    rng = np.random.default_rng(seed)
    rows = []

    for fraction in fractions:
        results = np.asarray(
            [
                evaluate(
                    perturb_weights(
                        base_weights,
                        fraction,
                        rng,
                        signed=signed,
                    )
                )
                for _ in range(n_repeats)
            ],
            dtype=float,
        )

        rows.append(
            {
                "scenario": f"+/-{int(round(fraction * 100))}%",
                "mean_rho": float(np.nanmean(results[:, 0])),
                "min_rho": (
                    float(np.nanmin(results[:, 1]))
                    if np.isfinite(results[:, 1]).any()
                    else float("nan")
                ),
                "top3_overlap": float(np.nanmean(results[:, 2])),
            }
        )

    equal_result = evaluate(
        equal_weight_stress_test(len(criteria))
    )

    rows.append(
        {
            "scenario": "Bobot sama",
            "mean_rho": equal_result[0],
            "min_rho": equal_result[1],
            "top3_overlap": equal_result[2],
        }
    )

    return pd.DataFrame(rows)
