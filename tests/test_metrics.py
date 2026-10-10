"""Uji dasar (sanity test) untuk modul-modul inti pipeline."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batik_fahp_topsis.fahp import (
    aggregate_group_matrix,
    conventional_fahp_weights,
    consistency_ratio,
    type2_fahp_weights,
)
from batik_fahp_topsis.pareto import compute_mean_ranks, pareto_efficient_set
from batik_fahp_topsis.topsis import equal_weight_stress_test, topsis_within_cluster
from batik_fahp_topsis.validity_metrics import partition_coefficient, partition_entropy


def _identity_consistent_matrix(n: int = 5) -> np.ndarray:
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:n]
    return weights[:, None] / weights[None, :]


def test_consistency_ratio_zero_for_consistent_matrix():
    """CR harus mendekati nol untuk matriks yang secara konstruksi konsisten sempurna."""
    matrix = _identity_consistent_matrix()
    result = consistency_ratio(matrix)
    assert result.cr < 1e-6
    assert result.consistent


def test_fahp_weights_sum_to_one():
    """Bobot FAHP Type-1 maupun Type-2 harus ternormalisasi (berjumlah 1) dan positif."""
    matrices = [_identity_consistent_matrix() for _ in range(3)]
    group = aggregate_group_matrix(matrices)
    w1 = conventional_fahp_weights(group)
    w2 = type2_fahp_weights(group)
    assert np.isclose(w1.sum(), 1.0)
    assert np.isclose(w2.sum(), 1.0)
    assert np.all(w1 > 0)
    assert np.all(w2 > 0)


def test_pareto_efficiency_and_mean_rank():
    """Konfigurasi yang terdominasi harus ditandai bukan Pareto-efficient, dan mean rank hanya
    dihitung di antara konfigurasi yang Pareto-efficient."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "n_features": [7, 7, 7],
            "k": [2, 5, 6],
            "SC": [0.3, 0.5, 0.4],
            "DBI": [1.2, 1.0, 0.9],
            "CHI": [100, 400, 420],
        }
    )
    pareto_df = pareto_efficient_set(df)
    # Baris 0 (K=2) terdominasi oleh baris 1 (K=5): sama buruk atau lebih buruk di ketiganya,
    # dan secara ketat lebih buruk pada sebagian.
    assert not pareto_df.loc[0, "pareto_efficient"]
    ranked = compute_mean_ranks(pareto_df)
    assert set(ranked["k"]) <= {5, 6}
    assert "mean_rank" in ranked.columns


def test_topsis_scores_in_unit_interval():
    """Skor TOPSIS harus selalu berada dalam rentang [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 4))
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    cost_mask = np.array([False, False, True, False])
    scores = topsis_within_cluster(X, weights, cost_mask)
    assert scores.shape == (20,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_equal_weight_stress_test_is_uniform():
    """Uji tekanan bobot-sama harus menghasilkan bobot yang seragam dan berjumlah 1."""
    w = equal_weight_stress_test(7)
    assert np.allclose(w, 1.0 / 7)
    assert np.isclose(w.sum(), 1.0)


def test_partition_coefficient_and_entropy_bounds():
    """PC harus berada di antara 1/c dan 1; PE harus non-negatif."""
    membership = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    pc = partition_coefficient(membership)
    pe = partition_entropy(membership)
    assert 1.0 / 2 <= pc <= 1.0
    assert pe >= 0.0
