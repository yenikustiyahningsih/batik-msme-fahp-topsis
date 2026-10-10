"""Kontrol seed acak terpusat (Reviewer 1, Komentar 7 / Komentar 8).

Setiap tahap yang bersifat stokastik dalam pipeline (inisialisasi K-Means, resampling
bootstrap, ablasi subset acak, inisialisasi baseline) mengambil seed-nya dari modul ini,
sehingga satu aturan yang tercatat mengatur reproduksibilitas di seluruh kode, bukan
di-set sembarangan per script.
"""
from __future__ import annotations

import numpy as np

MAIN_GRID_SEED = 42  # perbandingan grid gabungan full-data yang deterministik (Bagian 3.10)


def bootstrap_seed(repetition_index: int, base_seed: int = MAIN_GRID_SEED) -> int:
    """Seed yang deterministik dan berbeda untuk setiap repetisi bootstrap `repetition_index`.

    Eksperimen run-berulang sengaja menggunakan seed yang *berbeda* dari grid utama
    (manuskrip, Bagian 3.10), tetapi aturannya sendiri tetap harus reproduksibel:
    seed(i) = base_seed * 1000 + i.
    """
    if repetition_index < 0:
        raise ValueError("repetition_index harus >= 0")
    return base_seed * 1000 + repetition_index


def get_rng(seed: int = MAIN_GRID_SEED) -> np.random.Generator:
    """Mengembalikan numpy Generator dengan seed yang deterministik."""
    return np.random.default_rng(seed)
