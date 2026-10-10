"""Pembobotan kriteria Fuzzy AHP Type-2 dan konvensional (Type-1) (manuskrip Bagian 3.3).

Pipeline yang diimplementasikan di sini, mengikuti langkah-langkah manuskrip:

1. Masing-masing dari empat pakar memberikan matriks perbandingan berpasangan 16x16 yang crisp
   (skala Saaty 1-9).
2. Consistency Ratio (CR), diturunkan dari Consistency Index (CI) dan Random Index (RI),
   diperiksa untuk setiap matriks pakar *sebelum* fuzzifikasi (AHP consistency ratio, ref [32]),
   terhadap ambang batas konvensional 0.10.
3. Keempat matriks pakar (yang sudah konsisten) digabungkan secara element-wise melalui rata-rata
   geometrik menjadi satu matriks grup, yang CR-nya diperiksa ulang.
4. FAHP konvensional (Type-1): setiap judgment crisp hasil agregasi dipetakan ke bilangan fuzzy
   segitiga Type-1; bobot fuzzy konvensional diperoleh melalui metode extent-analysis / rata-rata
   geometrik dan dinormalisasi.
5. FAHP Type-2: setiap judgment crisp hasil agregasi dipetakan ke bilangan fuzzy segitiga
   Interval Type-2 (IT2) (upper membership function dan lower membership function yang lebih
   sempit di sekitar nilai crisp yang sama). Judgment pairwise IT2 diagregasi dengan rata-rata
   geometrik, direduksi tipenya (gaya centroid Karnik-Mendel), dan didefuzzifikasi menjadi skor
   crisp q_i per kriteria, lalu dinormalisasi agar bobot kriteria yang dipertahankan berjumlah
   satu.
6. Kriteria diurutkan menurun berdasarkan bobot FAHP Type-2; subset bertingkat berukuran N
   (N = 7, ..., 16) dibentuk dengan mempertahankan N kriteria berperingkat teratas (lihat
   `feature_selection.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59, 16: 1.605,
}


@dataclass
class ConsistencyResult:
    ci: float
    cr: float
    n: int
    consistent: bool


def consistency_ratio(pairwise_matrix: np.ndarray, threshold: float = 0.10) -> ConsistencyResult:
    """Consistency Index / Consistency Ratio ala Saaty untuk matriks perbandingan berpasangan yang crisp."""
    n = pairwise_matrix.shape[0]
    eigvals = np.linalg.eigvals(pairwise_matrix)
    lambda_max = float(np.max(eigvals.real))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, RANDOM_INDEX[16])
    cr = ci / ri if ri > 0 else 0.0
    return ConsistencyResult(ci=ci, cr=cr, n=n, consistent=cr <= threshold)


def aggregate_group_matrix(expert_matrices: list[np.ndarray]) -> np.ndarray:
    """Rata-rata geometrik element-wise dari E matriks perbandingan berpasangan crisp milik E pakar."""
    stacked = np.stack(expert_matrices, axis=0)
    return np.exp(np.mean(np.log(stacked), axis=0))


def _crisp_weights_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Estimasi bobot crisp ternormalisasi tanpa eigenvector utama, melalui metode rata-rata
    geometrik (row-product), dipakai sebagai vektor bobot acuan sebelum di-fuzzifikasi."""
    n = matrix.shape[0]
    row_geo_mean = np.prod(matrix, axis=1) ** (1.0 / n)
    return row_geo_mean / row_geo_mean.sum()


def conventional_fahp_weights(group_matrix: np.ndarray, fou_factor: float = 0.20) -> np.ndarray:
    """Bobot FAHP konvensional (Type-1).

    Setiap judgment crisp a_ij diperluas menjadi bilangan fuzzy segitiga simetris
    (a_ij * (1 - fou_factor), a_ij, a_ij * (1 + fou_factor)); bobot fuzzy diperoleh melalui
    metode rata-rata geometrik pada masing-masing matriks lower/middle/upper, lalu
    didefuzzifikasi dengan rata-rata ketiga komponen dan dinormalisasi ulang.
    """
    lower = group_matrix * (1 - fou_factor)
    upper = group_matrix * (1 + fou_factor)

    w_l = _crisp_weights_from_matrix(lower)
    w_m = _crisp_weights_from_matrix(group_matrix)
    w_u = _crisp_weights_from_matrix(upper)

    crisp = (w_l + w_m + w_u) / 3.0
    return crisp / crisp.sum()


def type2_fahp_weights(
    group_matrix: np.ndarray,
    umf_fou_factor: float = 0.20,
    lmf_fou_factor: float = 0.10,
) -> np.ndarray:
    """Bobot FAHP Interval Type-2.

    Setiap judgment crisp a_ij hasil agregasi direpresentasikan dengan bilangan fuzzy segitiga
    Interval Type-2, dengan footprint Upper Membership Function (UMF) sebesar `umf_fou_factor`
    dan footprint Lower Membership Function (LMF) yang lebih sempit sebesar `lmf_fou_factor` di
    sekitar nilai crisp yang sama, mengikuti representasi fuzzy Interval Type-2 yang dipakai pada
    model keputusan fuzzy Interval Type-2 (ref manuskrip [10, 11]).

    Reduksi tipe menggunakan centroid gaya Karnik-Mendel: vektor bobot rata-rata geometrik
    berbasis UMF dan berbasis LMF dirata-rata, lalu didefuzzifikasi dan dinormalisasi untuk
    menghasilkan skor crisp q_i per kriteria yang dipakai untuk perankingan dan pembentukan
    subset bertingkat.
    """
    umf_lower = group_matrix * (1 - umf_fou_factor)
    umf_upper = group_matrix * (1 + umf_fou_factor)
    lmf_lower = group_matrix * (1 - lmf_fou_factor)
    lmf_upper = group_matrix * (1 + lmf_fou_factor)

    w_umf = (_crisp_weights_from_matrix(umf_lower) + _crisp_weights_from_matrix(umf_upper)) / 2.0
    w_lmf = (_crisp_weights_from_matrix(lmf_lower) + _crisp_weights_from_matrix(lmf_upper)) / 2.0
    w_mid = _crisp_weights_from_matrix(group_matrix)

    # Reduksi tipe gaya centroid Karnik-Mendel pada estimasi bobot UMF/LMF/titik-tengah.
    q = (w_umf + w_lmf + 2.0 * w_mid) / 4.0
    return q / q.sum()


def footprint_of_uncertainty(w_umf: np.ndarray, w_lmf: np.ndarray) -> float:
    """Rata-rata lebar antara batas membership atas dan bawah (FOU), ukuran deskriptif sisa
    ketidakpastian Interval Type-2 yang dilaporkan bersama validitas khusus-fuzzy (Tabel 10)."""
    return float(np.mean(np.abs(w_umf - w_lmf)))


def rank_criteria(weights: np.ndarray, criteria_names: list[str]) -> list[str]:
    """Nama kriteria diurutkan menurun berdasarkan bobot (dasi/ties dipecah berdasarkan urutan asli)."""
    order = np.argsort(-weights, kind="stable")
    return [criteria_names[i] for i in order]
