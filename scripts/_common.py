"""Fungsi bantu bersama untuk script-script pipeline: pemuatan data, matriks pairwise pakar
contoh, dan utilitas plotting kecil. Bukan bagian dari API publik `batik_fahp_topsis`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from batik_fahp_topsis.fahp import consistency_ratio  # noqa: E402
from batik_fahp_topsis.preprocessing import ALL_CRITERIA  # noqa: E402

N_CRITERIA = len(ALL_CRITERIA)


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def example_expert_matrix(seed: int, n: int = N_CRITERIA, target_cr: float = 0.06) -> np.ndarray:
    """Membangun matriks perbandingan berpasangan crisp yang konsisten dan masuk akal, HANYA
    untuk keperluan demonstrasi. Pada pengajuan yang sebenarnya, ganti fungsi ini sepenuhnya
    dengan judgment asli dari keempat pakar (lihat params/fahp_pairwise_template.yaml).

    Konstruksi: menarik vektor "bobot sebenarnya" yang positif, membentuk matriks rasio w_i/w_j
    (yang secara konstruksi memiliki CR = 0), lalu menambahkan noise multiplikatif kecil dan
    membulatkannya ke nilai skala Saaty 1-9 (dan resiprokalnya) terdekat, yang menjaga CR tetap
    rendah tanpa persis nol.
    """
    rng = np.random.default_rng(seed)
    true_weights = rng.uniform(0.5, 5.0, size=n)
    ratio = true_weights[:, None] / true_weights[None, :]

    noise = rng.uniform(1 - 0.05, 1 + 0.05, size=(n, n))
    noisy = ratio * noise
    noisy = (noisy + 1.0 / noisy.T) / 2.0  # disimetrikan secara multiplikatif sebelum di-snap

    saaty_scale = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def snap(value: float) -> float:
        if value >= 1:
            return float(saaty_scale[np.argmin(np.abs(saaty_scale - value))])
        return 1.0 / snap(1.0 / value)

    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = snap(noisy[i, j])
            matrix[i, j] = v
            matrix[j, i] = 1.0 / v

    result = consistency_ratio(matrix)
    if not result.consistent:
        # Sangat tidak mungkin terjadi dengan konstruksi di atas, tapi tetap dijaga untuk
        # berjaga-jaga jika ada hasil acak yang kebetulan buruk.
        return example_expert_matrix(seed + 1000, n=n, target_cr=target_cr)
    return matrix


EXPERT_SEEDS = {
    "msme_office": 101,
    "msme_owner": 102,
    "batik_association": 103,
    "academic": 104,
}
