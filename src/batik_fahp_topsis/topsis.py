"""TOPSIS khusus per-cluster dan analisis sensitivitas (manuskrip Bagian 3.6, Tabel 12, Tabel 13).

Skor TOPSIS dihitung *di dalam* masing-masing cluster akhir secara terpisah (bukan secara
global), sehingga skor dari satu cluster tidak boleh dibandingkan langsung dengan skor dari
cluster lain -- pertanyaan ranking yang dijawab adalah "MSME mana di dalam cluster ini yang
harus diprioritaskan", bukan "MSME mana yang terbaik secara keseluruhan".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def topsis_within_cluster(
    X: np.ndarray,
    weights: np.ndarray,
    cost_mask: np.ndarray,
) -> np.ndarray:
    """TOPSIS standar pada data satu cluster.

    `X`: (n_members, n_criteria), sudah pada skala umum yang dinormalisasi Min-Max.
    `weights`: (n_criteria,) bobot kriteria FAHP Type-2 (atau yang sudah di-perturb) yang
    ternormalisasi.
    `cost_mask`: (n_criteria,) boolean, True jika kriteria tersebut berarah cost (makin
    rendah makin baik) -- dalam manuskrip ini, hanya C7 (Raw Material Price Increase) yang
    berarah cost.
    Mengembalikan closeness coefficient TOPSIS per baris, dalam rentang [0, 1], makin tinggi
    makin baik.
    """
    # Normalisasi vektor.
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    V = (X / norms) * weights[None, :]

    ideal_best = np.where(cost_mask, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(cost_mask, V.max(axis=0), V.min(axis=0))

    dist_best = np.sqrt(((V - ideal_best[None, :]) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((V - ideal_worst[None, :]) ** 2).sum(axis=1))

    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-12
    return dist_worst / denom


def cluster_specific_topsis(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    criteria: list[str],
    weights: np.ndarray,
    cost_criteria: list[str],
    id_column: str = "msme_id",
) -> pd.DataFrame:
    """Menjalankan `topsis_within_cluster` secara terpisah untuk setiap cluster pada `labels`,
    mengembalikan satu baris per MSME beserta cluster, skor TOPSIS, dan rank di dalam cluster-nya."""
    cost_mask = np.array([c in cost_criteria for c in criteria])
    out_rows = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        member_mask = labels == cluster_id
        X = df_scaled.loc[member_mask, criteria].to_numpy()
        scores = topsis_within_cluster(X, weights, cost_mask)
        sub = pd.DataFrame(
            {
                id_column: df_scaled.loc[member_mask, id_column].values,
                "cluster": cluster_id,
                "topsis_score": scores,
            }
        )
        sub["rank_within_cluster"] = sub["topsis_score"].rank(ascending=False, method="min").astype(int)
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True).sort_values(["cluster", "rank_within_cluster"])


def perturb_weights(
    weights: np.ndarray, fraction: float, rng: np.random.Generator, signed: bool = False
) -> np.ndarray:
    """Mengganggu (perturb) setiap bobot sebesar +/- `fraction` (mis. 0.05, 0.10, 0.15), lalu
    menormalisasi ulang agar berjumlah 1.

    signed=False: gangguan ~ Uniform(-fraction, +fraction) per bobot.
    signed=True : setiap bobot dikalikan (1 +/- fraction) dengan tanda acak, sehingga besar gangguan
                  selalu tepat `fraction` (skenario terburuk yang lebih ketat)."""
    if signed:
        noise = fraction * rng.choice([-1.0, 1.0], size=weights.shape)
    else:
        noise = rng.uniform(-fraction, fraction, size=weights.shape)
    perturbed = np.clip(weights * (1.0 + noise), 1e-6, None)
    return perturbed / perturbed.sum()


def equal_weight_stress_test(n_criteria: int) -> np.ndarray:
    """Uji tekanan bobot-sama (equal-weight stress test) yang meniadakan judgment pakar sama
    sekali (Bagian 4.9)."""
    return np.full(n_criteria, 1.0 / n_criteria)


def _per_cluster_stability(base_scores, new_scores, top_k):
    """Spearman rho dan overlap top-K dihitung DI DALAM tiap cluster (skor TOPSIS hanya bermakna
    dalam cluster), lalu dirata-rata/diminimumkan lintas cluster. Tie dipecah berdasarkan urutan baris."""
    from scipy.stats import spearmanr

    rhos, overlaps = [], []
    for b, p in zip(base_scores, new_scores):
        rhos.append(spearmanr(b, p)[0])
        k = min(top_k, len(b))
        top_b = set(np.argsort(-b, kind="stable")[:k])
        top_p = set(np.argsort(-p, kind="stable")[:k])
        overlaps.append(len(top_b & top_p) / k)
    return float(np.mean(rhos)), float(np.min(rhos)), float(np.mean(overlaps))


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
    """Tabel 13: untuk setiap persentase gangguan (dan uji tekanan bobot-sama), mengganggu bobot
    berulang kali, menghitung ulang TOPSIS per-cluster, lalu meringkas -- PER CLUSTER -- korelasi
    Spearman dan overlap top-K terhadap ranking bobot-dasar.

    mean_rho  = rata-rata rho lintas cluster dan ulangan; min_rho = rho terendah dari semua cluster
    dan ulangan (kasus terburuk); top3_overlap = rata-rata overlap top-3 per cluster."""
    cost_mask = np.array([c in cost_criteria for c in criteria])
    clusters = [c for c in sorted(set(labels)) if c != -1]
    Xs = [df_scaled.loc[labels == c, criteria].to_numpy() for c in clusters]
    base = [topsis_within_cluster(X, base_weights, cost_mask) for X in Xs]

    def one(w):
        return _per_cluster_stability(base, [topsis_within_cluster(X, w, cost_mask) for X in Xs], top_k)

    rng = np.random.default_rng(seed)
    rows = []
    for f in fractions:
        res = np.array([one(perturb_weights(base_weights, f, rng, signed=signed)) for _ in range(n_repeats)])
        rows.append({"scenario": f"+/-{int(round(f * 100))}%", "mean_rho": res[:, 0].mean(),
                     "min_rho": res[:, 1].min(), "top3_overlap": res[:, 2].mean()})
    r = one(equal_weight_stress_test(len(criteria)))
    rows.append({"scenario": "Bobot sama", "mean_rho": r[0], "min_rho": r[1], "top3_overlap": r[2]})
    return pd.DataFrame(rows)
