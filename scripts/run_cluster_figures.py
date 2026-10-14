"""Tahap 13: Figure 3 (trade-off kandidat K), Figure 4 (profil centroid ternormalisasi) dan Figure 5 (distribusi UMKM per cluster)
untuk konfigurasi akhir yang terpilih. Keduanya dihasilkan dari file hasil yang sama dengan tabel
lain, sehingga angka di gambar identik dengan Tabel 12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.clustering import fit_kmeans

from _figstyle import DPI, FIG_W, FONT, TITLE_FONT, wrap

NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
       10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen"}


def plot_figure_03(results: pd.DataFrame, n_features: int, selected_k: int, outdir: Path,
                   dpi: int = DPI, font: int = FONT) -> None:
    """Figure 3: trade-off DBI (sumbu x) dan SC (sumbu y) antar kandidat K pada jumlah fitur
    terpilih. Konfigurasi akhir ditandai bintang merah."""
    sub = results[results["n_features"] == n_features].sort_values("k")
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1))
    cmap = plt.get_cmap("tab10")
    for _, row in sub.iterrows():
        k = int(row["k"])
        if k == selected_k:
            ax.scatter(row["DBI"], row["SC"], s=170, marker="*", color="#d62728", zorder=3)
        else:
            ax.scatter(row["DBI"], row["SC"], s=55, color=cmap((k - 2) % 10), zorder=3)
        ax.annotate(f"K={k}", (row["DBI"], row["SC"]), textcoords="offset points", xytext=(4, 5), fontsize=font)
    ax.set_title(wrap(f"Trade-off among {NUM[n_features]}-feature K candidates"), fontsize=TITLE_FONT)
    ax.set_xlabel("Davies-Bouldin Index (lower is better)", fontsize=font)
    ax.set_ylabel("Silhouette Coefficient (higher is better)", fontsize=font)
    ax.tick_params(labelsize=font)
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "figure_03_tradeoff.png", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--results", default="outputs/results_90_configs.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT})

    df_scaled = load_data(args.scaled_data)
    cfg = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = cfg["features"], int(cfg["k"])
    X = df_scaled[features].to_numpy()
    _, labels = fit_kmeans(X, k)  # seed dan pengaturan sama dengan tahap pemilihan konfigurasi

    plot_figure_03(load_data(args.results), int(cfg["n_features"]), k, outdir)

    # Figure 4: rata-rata (centroid) setiap fitur terpilih per cluster, pada skala Min-Max [0, 1].
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in range(k)])
    # Cluster pada sumbu x dan fitur pada sumbu y, supaya angka 10 pt muat pada lebar satu kolom.
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    im = ax.imshow(centroids.T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(k), [str(c + 1) for c in range(k)])
    ax.set_yticks(range(len(features)), features)
    ax.set_xlabel("Cluster")
    mid = (centroids.min() + centroids.max()) / 2
    for j in range(k):
        for i in range(len(features)):
            ax.text(j, i, f"{centroids[j, i]:.2f}", ha="center", va="center", fontsize=FONT,
                    color="black" if centroids[j, i] > mid else "white")
    ax.set_title(wrap(f"Normalized centroid profiles of the {NUM[k]} final clusters"), fontsize=TITLE_FONT)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized centroid", fontsize=FONT)
    fig.tight_layout()
    fig.savefig(outdir / "figure_04_centroid_profiles.png", dpi=DPI)
    plt.close(fig)

    # Figure 5: jumlah UMKM per cluster.
    counts = np.bincount(labels, minlength=k)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9))
    bars = ax.bar(range(1, k + 1), counts, color="#1f77b4")
    ax.bar_label(bars, fontsize=FONT)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xticks(range(1, k + 1))
    ax.set_title(wrap(f"Distribution of MSMEs across the final {NUM[k]} clusters"), fontsize=TITLE_FONT)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of MSMEs")
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "figure_05_cluster_distribution.png", dpi=DPI)
    plt.close(fig)

    print(f"Menulis figure_03_tradeoff.png, figure_04_centroid_profiles.png dan figure_05_cluster_distribution.png "
          f"(K={k}, fitur={features}; ukuran cluster {counts.tolist()})")


if __name__ == "__main__":
    main()