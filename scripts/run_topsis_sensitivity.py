
"""Tahap 11: analisis sensitivitas bobot TOPSIS -> Tabel 14, Figure 6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_data
from _figstyle import DPI, FIG_W, FONT, TITLE_FONT, wrap
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.topsis import sensitivity_summary

COST_CRITERIA = ["C7"]


def plot_figure_06(
    table,
    outdir: Path,
    dpi: int = DPI,
    font: int = FONT,
) -> None:
    """Figure 6: stabilitas ranking TOPSIS berdasarkan analisis sensitivitas."""
    labels = [
        str(value)
        .replace("+/-", "\u00b1")
        .replace("Bobot sama", "Equal\nweights")
        for value in table["scenario"]
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.9))
    ax.plot(
        labels,
        table["mean_rho"],
        marker="o",
        label="Spearman rho",
    )
    ax.plot(
        labels,
        table["top3_overlap"],
        marker="s",
        label="Top-3 Jaccard similarity",
    )

    ax.set_title(
        wrap("Stability of TOPSIS priorities under weight perturbation"),
        fontsize=TITLE_FONT,
    )
    ax.set_ylabel("Stability", fontsize=font)
    ax.tick_params(labelsize=font)
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    ax.legend(fontsize=font, loc="best")

    # Batas sumbu dinamis agar nilai hasil tidak terpotong.
    plotted_values = np.concatenate(
        [
            table["mean_rho"].to_numpy(dtype=float),
            table["top3_overlap"].to_numpy(dtype=float),
        ]
    )
    plotted_values = plotted_values[np.isfinite(plotted_values)]

    if len(plotted_values):
        lower = min(0.0, float(plotted_values.min()) - 0.05)
        upper = min(1.05, max(1.0, float(plotted_values.max()) + 0.05))
        ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(
        outdir / "figure_06_topsis_sensitivity.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analisis sensitivitas bobot TOPSIS per cluster."
    )
    parser.add_argument(
        "--scaled-data",
        default="outputs/preprocessed_scaled.csv",
    )
    parser.add_argument(
        "--selected-config",
        default="outputs/selected_configuration.json",
    )
    parser.add_argument(
        "--fahp-weights",
        default="outputs/table_03_fahp_weights.csv",
    )
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--topsis-data",
        default=None,
        help=(
            "CSV indikator untuk matriks keputusan TOPSIS. "
            "Pastikan skala data sesuai metodologi manuskrip. "
            "Tanpa opsi ini, data Min-Max digunakan."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Muat data clustering.
    df_scaled = load_data(args.scaled_data)

    if "msme_id" not in df_scaled.columns:
        raise ValueError(
            "Kolom 'msme_id' tidak ditemukan pada data clustering."
        )
    if df_scaled["msme_id"].duplicated().any():
        raise ValueError(
            "Terdapat msme_id duplikat pada data clustering."
        )

    config_path = Path(args.selected_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    features = config["features"]
    k = int(config["k"])

    if not features:
        raise ValueError("Konfigurasi tidak memiliki fitur.")
    if k < 2:
        raise ValueError("Jumlah cluster K harus minimal 2.")

    missing_scaled = set(features) - set(df_scaled.columns)
    if missing_scaled:
        raise ValueError(
            f"Fitur tidak ditemukan pada data clustering: "
            f"{sorted(missing_scaled)}"
        )

    X = df_scaled[features].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Data clustering mengandung NaN atau nilai tak hingga."
        )

    # Muat dan validasi bobot FAHP.
    weights_table = load_data(args.fahp_weights)

    required_weight_columns = {
        "criterion",
        "type2_fahp_weight",
    }
    missing_weight_columns = (
        required_weight_columns - set(weights_table.columns)
    )
    if missing_weight_columns:
        raise ValueError(
            "Kolom tabel bobot tidak ditemukan: "
            f"{sorted(missing_weight_columns)}"
        )

    if weights_table["criterion"].duplicated().any():
        raise ValueError(
            "Terdapat kriteria duplikat pada tabel bobot FAHP."
        )

    weights_table = weights_table.set_index("criterion")

    missing_weights = set(features) - set(weights_table.index)
    if missing_weights:
        raise ValueError(
            f"Bobot tidak tersedia untuk fitur: {sorted(missing_weights)}"
        )

    w = weights_table.loc[
        features, "type2_fahp_weight"
    ].to_numpy(dtype=float)

    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Bobot FAHP tidak valid.")

    w = w / w.sum()

    # Clustering tetap menggunakan data Min-Max.
    _, labels = fit_kmeans(X, k)

    # Tentukan matriks keputusan TOPSIS.
    df_topsis = df_scaled

    if args.topsis_data:
        df_topsis = load_data(args.topsis_data)

        if "msme_id" not in df_topsis.columns:
            raise ValueError(
                "Kolom 'msme_id' tidak ditemukan pada data TOPSIS."
            )
        if df_topsis["msme_id"].duplicated().any():
            raise ValueError(
                "Terdapat msme_id duplikat pada data TOPSIS."
            )
        if list(df_topsis["msme_id"]) != list(df_scaled["msme_id"]):
            raise ValueError(
                "ID pada --topsis-data tidak sama atau urutannya "
                "berbeda dari data clustering."
            )

        missing_topsis = set(features) - set(df_topsis.columns)
        if missing_topsis:
            raise ValueError(
                f"Fitur tidak ditemukan pada data TOPSIS: "
                f"{sorted(missing_topsis)}"
            )

        X_topsis = df_topsis[features].to_numpy(dtype=float)
        if not np.isfinite(X_topsis).all():
            raise ValueError(
                "Data TOPSIS mengandung NaN atau nilai tak hingga."
            )

    # Hitung analisis sensitivitas.
    table14 = sensitivity_summary(
        df_scaled=df_topsis,
        labels=labels,
        criteria=features,
        base_weights=w,
        cost_criteria=COST_CRITERIA,
    )

    csv_path = outdir / "table_14_topsis_sensitivity.csv"
    table14.to_csv(csv_path, index=False)

    plot_figure_06(table14, outdir)

    print(f"Konfigurasi: K={k}, jumlah fitur={len(features)}")
    print(f"Tabel sensitivitas: {csv_path}")
    print(
        "Grafik sensitivitas: "
        f"{outdir / 'figure_06_topsis_sensitivity.png'}"
    )
    print(table14.to_string(index=False))


if __name__ == "__main__":
    main()
