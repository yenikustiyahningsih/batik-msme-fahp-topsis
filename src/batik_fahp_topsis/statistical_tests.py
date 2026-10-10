"""Uji signifikansi statistik (manuskrip Bagian 3.5, Tabel 8).

Uji omnibus Friedman di antara metode-metode pada 30 pasangan hasil bootstrap, diikuti dengan
uji pairwise Wilcoxon signed-rank yang dikoreksi Holm (metode usulan vs. setiap alternatif) dan
ukuran efek rank-biserial correlation berpasangan, dijalankan terpisah untuk SC, DBI, dan CHI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def friedman_test(paired_scores: pd.DataFrame) -> dict:
    """`paired_scores`: baris = repetisi bootstrap, kolom = metode, untuk satu metrik."""
    stat, p = stats.friedmanchisquare(*[paired_scores[c].values for c in paired_scores.columns])
    return {"chi_square": float(stat), "p_value": float(p), "df": paired_scores.shape[1] - 1}


def rank_biserial_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Korelasi rank-biserial berpasangan (matched-pairs), dihitung dari statistik signed-rank."""
    diffs = np.asarray(x) - np.asarray(y)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    n_total = r_plus + r_minus
    return float((r_plus - r_minus) / n_total) if n_total > 0 else 0.0


def holm_wilcoxon(paired_scores: pd.DataFrame, reference_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """Uji Wilcoxon signed-rank antara `reference_col` dan setiap kolom lainnya, dengan koreksi
    step-down Holm untuk family-wise error rate, ditambah ukuran efek rank-biserial per pembanding."""
    comparators = [c for c in paired_scores.columns if c != reference_col]
    raw_p = []
    effect = []
    for c in comparators:
        diffs = np.asarray(paired_scores[reference_col]) - np.asarray(paired_scores[c])
        if np.all(diffs == 0):
            # Kedua metode menghasilkan nilai identik pada semua repetisi (mis. pada data contoh
            # atau saat bobot Type-1 = Type-2): tidak ada perbedaan yang dapat diuji, p = 1.
            p = 1.0
        else:
            stat, p = stats.wilcoxon(paired_scores[reference_col], paired_scores[c])
        raw_p.append(p)
        effect.append(rank_biserial_effect_size(paired_scores[reference_col], paired_scores[c]))

    order = np.argsort(raw_p)
    m = len(raw_p)
    holm_p = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * raw_p[idx]
        running_max = max(running_max, adj)
        holm_p[idx] = min(running_max, 1.0)

    return pd.DataFrame(
        {
            "comparator": comparators,
            "raw_p_value": raw_p,
            "holm_adjusted_p_value": holm_p,
            "rank_biserial_effect_size": effect,
            "significant_at_alpha": holm_p <= alpha,
        }
    )
