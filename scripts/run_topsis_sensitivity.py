"""Tahap 11: analisis sensitivitas bobot TOPSIS -> Tabel 13, Figure 7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_data
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.topsis import sensitivity_summary

COST_CRITERIA = ["C7"]


from _figstyle import DPI, FIG_W, FONT, TITLE_FONT, wrap


def plot_figure_06(table, outdir: Path, dpi: int = DPI, font: int = FONT) -> None:
    """Figure 6: stabilitas prioritas TOPSIS (dengan judul pendek). Dibuat dari tabel yang sama dengan
    Tabel 14, sehingga nilai gambar identik dengan tabel."""
    labels = [str(x).replace("+/-", "\u00b1").replace("Bobot sama", "Equal\nweights") for x in table["scenario"]]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9))
    ax.plot(labels, table["mean_rho"], marker="o", label="Spearman rho")
    ax.plot(labels, table["top3_overlap"], marker="s", label="Top-3 overlap")
    ax.set_title(wrap("Stability of TOPSIS priorities under weight perturbation"), fontsize=TITLE_FONT)
    ax.set_ylabel("Stability", fontsize=font)
    ax.tick_params(labelsize=font)
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    ax.set_ylim(0.78, 1.02)
    ax.legend(fontsize=font, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "figure_06_topsis_sensitivity.png", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--fahp-weights", default="outputs/table_03_fahp_weights.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--topsis-data",
        default=None,
        help="CSV indikator pada SKALA ASLI (mis. data/batik_imputed_frozen.csv) untuk matriks keputusan "
        "TOPSIS. Naskah Tabel 12-13 memakai skala asli (1-4); tanpa opsi ini dipakai data Min-Max.",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = config["features"], config["k"]
    X = df_scaled[features].to_numpy()

    weights_table = load_data(args.fahp_weights).set_index("criterion")
    w = weights_table.loc[features, "type2_fahp_weight"].to_numpy()
    w = w / w.sum()

    _, labels = fit_kmeans(X, k)  # clustering tetap pada data Min-Max
    df_topsis = df_scaled
    if args.topsis_data:
        df_topsis = load_data(args.topsis_data)
        assert list(df_topsis["msme_id"]) == list(df_scaled["msme_id"]), "ID --topsis-data tidak sama"
    table13 = sensitivity_summary(df_topsis, labels, features, w, COST_CRITERIA)
    table13.to_csv(outdir / "table_14_topsis_sensitivity.csv", index=False)

    plot_figure_06(table13, outdir)

    print(table13.to_string(index=False))


if __name__ == "__main__":
    main()