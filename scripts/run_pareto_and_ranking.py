"""Tahap 4: efisiensi Pareto + seleksi mean-rank -> Tabel 5, Tabel 6 (Figure 3 dibuat di run_cluster_figures.py).

Inilah kode yang dirujuk oleh caption/teks Tabel 5 versi revisi pada manuskrip: mean rank
selalu dihitung atas populasi Pareto-efficient yang *lengkap*, dan subset "leading" yang
dicetak adalah langkah tampilan terpisah yang diberi label jelas (Reviewer 1, Komentar 5 dan 6).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import load_data
from batik_fahp_topsis.pareto import compute_mean_ranks, leading_subset, pareto_efficient_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results_90_configs.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--leading-n", type=int, default=8)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = load_data(args.results)
    pareto_df = pareto_efficient_set(grid)
    ranked = compute_mean_ranks(pareto_df)  # ranking selalu atas SELURUH himpunan Pareto-efficient

    # Tabel 6: setiap konfigurasi Pareto-efficient beserta mean rank-nya.
    ranked.to_csv(outdir / "table_07_pareto_bootstrap.csv", index=False)

    # Tabel 5: subset leading (mean-rank terendah), hanya untuk ditampilkan.
    leading = leading_subset(ranked, n=args.leading_n)
    leading.to_csv(outdir / "table_06_leading_configs.csv", index=False)

    # Figure 3 dibuat di run_cluster_figures.py, karena membutuhkan konfigurasi akhir yang terpilih.

    best = ranked.iloc[0]
    print(f"Konfigurasi Pareto-efficient dengan mean-rank terendah: N={int(best['n_features'])}, "
          f"K={int(best['k'])}, mean_rank={best['mean_rank']:.2f}")
    print("CATATAN: pastikan ini juga terhadap stabilitas bootstrap dan kelayakan ukuran cluster "
          "(lihat run_bootstrap_stability.py) sebelum menganggapnya sebagai seleksi final -- "
          "lihat pareto.select_final_configuration().")


if __name__ == "__main__":
    main()