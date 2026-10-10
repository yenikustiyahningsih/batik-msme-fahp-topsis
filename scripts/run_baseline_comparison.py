"""Tahap 6: perbandingan baseline pada representasi fitur dan K yang sudah terpilih -> Tabel 7.

Sesuai teks revisi Bagian 4.3 pada manuskrip, ini mengevaluasi mekanisme partisi alternatif pada
representasi fitur dan jumlah cluster yang SUDAH dipilih oleh pipeline usulan; ini BUKAN
perbandingan yang di-tuning secara independen (Reviewer 2, Komentar 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import clara, fcm, it2fcm, load_baseline_params, weighted_kmeans
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.validity_metrics import internal_validity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--fahp-weights", default="outputs/table_03_fahp_weights.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = config["features"], config["k"]
    X = df_scaled[features].to_numpy()

    weights_table = load_data(args.fahp_weights).set_index("criterion")
    type2_w = weights_table.loc[features, "type2_fahp_weight"].to_numpy()
    type2_w = type2_w / type2_w.sum()
    type1_w = weights_table.loc[features, "type1_fahp_weight"].to_numpy()
    type1_w = type1_w / type1_w.sum()

    params = load_baseline_params()
    rows = []

    _, labels_proposed = fit_kmeans(X, k, seed=params["global"]["random_state"])
    m = internal_validity(X, labels_proposed)
    rows.append({"method": "Proposed", **m})

    labels_clara, _ = clara(X, k, params)
    rows.append({"method": "CLARA", **internal_validity(X, labels_clara)})

    fcm_result = fcm(X, k, params)
    rows.append({"method": "FCM", **internal_validity(X, fcm_result["labels"])})

    it2fcm_result = it2fcm(X, k, params)
    rows.append({"method": "IT2FCM", **internal_validity(X, it2fcm_result["labels"])})

    labels_wk_fahp, _ = weighted_kmeans(X, type1_w, k, params)
    rows.append({"method": "Weighted K-Means (FAHP)", **internal_validity(X, labels_wk_fahp)})

    labels_wk_t2, _ = weighted_kmeans(X, type2_w, k, params)
    rows.append({"method": "Weighted K-Means (Type-2)", **internal_validity(X, labels_wk_t2)})

    table7 = pd.DataFrame(rows)
    table7.to_csv(outdir / "table_08_baseline_comparison.csv", index=False)
    print(table7.to_string(index=False))


if __name__ == "__main__":
    main()
