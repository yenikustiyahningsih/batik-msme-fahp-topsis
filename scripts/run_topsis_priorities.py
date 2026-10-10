"""Tahap 10: prioritas TOPSIS khusus per-cluster -> Tabel 12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_data
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.topsis import cluster_specific_topsis

COST_CRITERIA = ["C7"]  # Raw Material Price Increase: satu-satunya kriteria berarah cost.


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
    parser.add_argument("--top-k", type=int, default=3)
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
    priorities = cluster_specific_topsis(df_topsis, labels, features, w, COST_CRITERIA)

    names_path = Path(__file__).resolve().parents[1] / "data" / "pemetaan_id_nama_usaha.csv"
    if names_path.exists():
        priorities = priorities.merge(load_data(names_path), on="msme_id", how="left")
    n_tied = int((priorities.groupby("cluster")["rank_within_cluster"].transform(
        lambda r: r.map(r.value_counts())) > 1).sum())
    print(f"Jumlah UMKM dengan skor kembar (tie) dalam cluster-nya: {n_tied}")
    top_k = priorities[priorities["rank_within_cluster"] <= args.top_k]
    top_k.to_csv(outdir / "table_13_topsis_priorities.csv", index=False)
    priorities.to_csv(outdir / "topsis_priorities_full.csv", index=False)
    print(top_k.to_string(index=False))


if __name__ == "__main__":
    main()
