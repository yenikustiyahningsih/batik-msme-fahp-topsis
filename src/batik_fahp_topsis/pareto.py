"""Efisiensi Pareto dan seleksi mean-rank berbobot sama (manuskrip Bagian 3.4, Algoritma 1,
Langkah 3; menjawab Komentar 5 dan 6 Reviewer 1).

PENTING (populasi mean-rank, yang disoroti Reviewer 1): kolom "Mean rank" adalah rank ordinal
berbobot-sama setiap konfigurasi pada SC, DBI, dan CHI yang dihitung di dalam **seluruh himpunan
konfigurasi Pareto-efficient** yang teridentifikasi dari grid 90-konfigurasi penuh -- bukan
rank ulang yang dibatasi pada subset baris yang kebetulan dicetak di sebuah tabel. `compute_mean_ranks`
di bawah ini SELALU melakukan ranking atas populasi Pareto-efficient yang lengkap yang diberikan
kepadanya; `leading_subset` adalah langkah terpisah yang murni untuk keperluan tampilan, yaitu
memilih baris dengan mean-rank terendah untuk ditampilkan, dan tidak boleh pernah dipakai sebagai
populasi ranking itu sendiri. Pemisahan ini persis seperti yang sekarang dinyatakan secara eksplisit
pada caption dan teks di sekitar Tabel 5 dalam manuskrip, dan di sini ditegakkan lewat kode (bukan
dibiarkan direproduksi secara manual) agar keduanya tidak bisa diam-diam menyimpang lagi.
"""
from __future__ import annotations

import pandas as pd


def is_dominated(row: pd.Series, others: pd.DataFrame) -> bool:
    """Suatu konfigurasi dikatakan terdominasi jika ada konfigurasi lain yang tidak lebih buruk
    pada SC, DBI, dan CHI, dan secara ketat lebih baik pada setidaknya satu di antaranya
    (SC/CHI makin tinggi makin baik, DBI makin rendah makin baik)."""
    no_worse = (others["SC"] >= row["SC"]) & (others["DBI"] <= row["DBI"]) & (others["CHI"] >= row["CHI"])
    strictly_better = (others["SC"] > row["SC"]) | (others["DBI"] < row["DBI"]) | (others["CHI"] > row["CHI"])
    return bool((no_worse & strictly_better).any())


def pareto_efficient_set(grid_results: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan subset dari `grid_results` (hasil `clustering.results_to_dataframe`)
    yang Pareto-efficient pada (SC, DBI, CHI), dengan kolom boolean tambahan `pareto_efficient`."""
    df = grid_results.copy().reset_index(drop=True)
    flags = []
    for i, row in df.iterrows():
        others = df.drop(index=i)
        flags.append(not is_dominated(row, others))
    df["pareto_efficient"] = flags
    return df


def compute_mean_ranks(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Mean rank berbobot-sama atas SC, DBI, dan CHI, dihitung di dalam seluruh populasi
    Pareto-efficient (standard competition ranking: baris yang seri berbagi rank terendah dalam
    kelompok yang seri tersebut)."""
    df = pareto_df[pareto_df["pareto_efficient"]].copy()
    df["rank_sc"] = df["SC"].rank(ascending=False, method="min")
    df["rank_dbi"] = df["DBI"].rank(ascending=True, method="min")
    df["rank_chi"] = df["CHI"].rank(ascending=False, method="min")
    df["mean_rank"] = (df["rank_sc"] + df["rank_dbi"] + df["rank_chi"]) / 3.0
    return df.sort_values("mean_rank")


def leading_subset(ranked_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """`n` baris dengan mean-rank terendah, *hanya untuk ditampilkan* (manuskrip Tabel 5). Jangan
    pernah memasukkan subset ini kembali ke `compute_mean_ranks` -- rank harus selalu dihitung
    atas seluruh populasi Pareto-efficient, persis seperti yang sudah dilakukan `compute_mean_ranks`."""
    return ranked_df.head(n)


def select_final_configuration(
    ranked_df: pd.DataFrame,
    bootstrap_ari: dict[tuple[int, int], float],
    min_cluster_fraction: float = 0.03,
    cluster_sizes: dict[tuple[int, int], list[int]] | None = None,
) -> pd.Series:
    """Menerapkan hierarki seleksi yang dinyatakan dalam manuskrip (Bagian 3.4 / Algoritma 1,
    Langkah 3):
    1. Pareto-efficient,
    2. mean-rank terendah sebagai kriteria utama,
    3. stabilitas resampling (bootstrap ARI) yang tinggi,
    4. ukuran cluster yang praktis dipakai (tidak ada cluster di bawah `min_cluster_fraction`
       dari jumlah sampel).

    Fungsi ini sengaja dibuat konservatif: tidak ada "K = 5" yang di-hard-code di mana pun.
    Jalankan ulang fungsi ini terhadap `outputs/results_90_configs.csv` (hasil
    `scripts/run_joint_optimization.py`) untuk memastikan konfigurasi mana yang benar-benar
    dipilih oleh hierarki ini untuk suatu dataset tertentu, termasuk data MSME asli (non-sintetis)
    begitu sudah tersedia.
    """
    candidates = ranked_df.sort_values("mean_rank")
    for _, row in candidates.iterrows():
        key = (int(row["n_features"]), int(row["k"]))
        ari = bootstrap_ari.get(key)
        if ari is None or ari < 0.90:
            continue
        if cluster_sizes is not None:
            sizes = cluster_sizes.get(key)
            if sizes is not None:
                total = sum(sizes)
                if any((s / total) < min_cluster_fraction for s in sizes):
                    continue
        return row
    # Jika tidak ada kandidat yang lolos semua filter, kembalikan konfigurasi dengan mean-rank
    # terendah tunggal sebagai fallback.
    return candidates.iloc[0]
