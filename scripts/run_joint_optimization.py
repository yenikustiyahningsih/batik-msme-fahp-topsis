"""Tahap 3: grid search gabungan fitur-cluster 90-konfigurasi -> results_90_configs.csv,
Figure 2. Ini adalah satu-satunya file hasil otoritatif yang dibaca oleh semua tahap berikutnya."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.clustering import results_to_dataframe, run_joint_grid
from batik_fahp_topsis.feature_selection import nested_fahp_subsets


def composite_score(df: pd.DataFrame) -> pd.Series:
    """Skor komposit ternormalisasi (0-1), dipakai hanya sebagai ukuran pendukung deskriptif --
    tidak pernah sebagai kriteria seleksi utama (manuskrip Bagian 3.5)."""
    sc_n = (df["SC"] - df["SC"].min()) / (df["SC"].max() - df["SC"].min())
    dbi_n = 1 - (df["DBI"] - df["DBI"].min()) / (df["DBI"].max() - df["DBI"].min())
    chi_n = (df["CHI"] - df["CHI"].min()) / (df["CHI"].max() - df["CHI"].min())
    return (sc_n + dbi_n + chi_n) / 3.0


from _figstyle import DPI, FIG_W, FONT, TITLE_FONT, wrap


def plot_figure_02(results_df, outdir: Path, dpi: int = DPI, font: int = FONT) -> None:
    """Figure 2: heatmap skor komposit ternormalisasi di seluruh 90 konfigurasi (judul pendek di dalam gambar). Nilai pada baris fitur paling sedikit dicetak di dalam sel."""
    pivot = results_df.pivot(index="n_features", columns="k", values="composite_score")
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=font)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=font)
    ax.set_title(wrap("Normalized composite score across 90 configurations"), fontsize=TITLE_FONT)
    ax.set_xlabel("Number of clusters K", fontsize=font)
    ax.set_ylabel("Number of retained features", fontsize=font)
    if FIG_W >= 5.5:  # pada lebar satu kolom, angka 10 pt akan saling menimpa
        for j, value in enumerate(pivot.values[0]):
            ax.text(j, 0, f"{value:.2f}", ha="center", va="center", fontsize=font, color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Composite score", fontsize=font)
    cbar.ax.tick_params(labelsize=font)
    fig.tight_layout()
    fig.savefig(outdir / "figure_02_composite_score.png", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--ranked-criteria", default="outputs/fahp_ranked_criteria.txt")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    ranked_criteria = Path(args.ranked_criteria).read_text(encoding="utf-8").strip().split(",")

    nested = nested_fahp_subsets(ranked_criteria, n_values=range(7, 17))
    results = run_joint_grid(df_scaled, nested, k_values=range(2, 11))
    results_df = results_to_dataframe(results)
    results_df["composite_score"] = composite_score(results_df)
    results_df.to_csv(outdir / "results_90_configs.csv", index=False)

    plot_figure_02(results_df, outdir)

    print(f"Setiap konfigurasi N/K Pareto-efficient ada di {outdir / 'results_90_configs.csv'}")


if __name__ == "__main__":
    main()