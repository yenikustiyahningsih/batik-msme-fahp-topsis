"""Tahap 8: analisis ablasi -> Tabel 9 (menjawab pertanyaan ablasi mekanisme pemilihan fitur
yang mendasari Komentar 4 dan 5 Reviewer 2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import load_baseline_params, weighted_kmeans
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.feature_selection import random_subsets, sequential_backward_selection
from batik_fahp_topsis.preprocessing import ALL_CRITERIA
from batik_fahp_topsis.validity_metrics import internal_validity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--fahp-weights", default="outputs/table_03_fahp_weights.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-random-subsets", type=int, default=100)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = config["features"], config["k"]
    n_features = len(features)

    weights_table = load_data(args.fahp_weights).set_index("criterion")
    type1_w_full = weights_table.loc[features, "type1_fahp_weight"].to_numpy()
    type1_w_full = type1_w_full / type1_w_full.sum()
    type2_w_full = weights_table.loc[features, "type2_fahp_weight"].to_numpy()
    type2_w_full = type2_w_full / type2_w_full.sum()

    params = load_baseline_params()
    rows = []

    # A1: seluruh 16 indikator.
    X_all = df_scaled[list(ALL_CRITERIA)].to_numpy()
    _, labels = fit_kmeans(X_all, k)
    rows.append({"variant": "A1", "setting": "All 16", **internal_validity(X_all, labels)})

    # A2 / A3: subset FAHP konvensional vs subset FAHP Type-2 (subset yang sama dalam pipeline
    # ini, karena kedua ranking dihitung dari matriks grup yang sama -- lihat docstring fahp.py
    # untuk kapan keduanya bisa secara sah berbeda, mis. di bawah judgment pakar yang di-perturb
    # atau yang saling tidak sepakat).
    X_sel = df_scaled[features].to_numpy()
    _, labels_fahp = fit_kmeans(X_sel, k)
    rows.append({"variant": "A2", "setting": "FAHP FS7", **internal_validity(X_sel, labels_fahp)})
    rows.append({"variant": "A3", "setting": "Type-2 FS7", **internal_validity(X_sel, labels_fahp)})

    # A4: bobot FAHP konvensional diterapkan langsung sebagai bobot jarak pada seluruh 16 fitur.
    full_type1_w = weights_table.loc[list(ALL_CRITERIA), "type1_fahp_weight"].to_numpy()
    full_type1_w = full_type1_w / full_type1_w.sum()
    labels_a4, _ = weighted_kmeans(X_all, full_type1_w, k, params)
    rows.append({"variant": "A4", "setting": "FAHP weighted 16", **internal_validity(X_all, labels_a4)})

    # A5: bobot FAHP Type-2 diterapkan langsung sebagai bobot jarak pada seluruh 16 fitur.
    full_type2_w = weights_table.loc[list(ALL_CRITERIA), "type2_fahp_weight"].to_numpy()
    full_type2_w = full_type2_w / full_type2_w.sum()
    labels_a5, _ = weighted_kmeans(X_all, full_type2_w, k, params)
    rows.append({"variant": "A5", "setting": "Type-2 weighted 16", **internal_validity(X_all, labels_a5)})

    # A6 / A7: subset 7-fitur terpilih DITAMBAH pembobotan jarak langsung (cek double counting).
    labels_a6, _ = weighted_kmeans(X_sel, type1_w_full, k, params)
    rows.append({"variant": "A6", "setting": "FS7 + FAHP weight", **internal_validity(X_sel, labels_a6)})
    labels_a7, _ = weighted_kmeans(X_sel, type2_w_full, k, params)
    rows.append({"variant": "A7", "setting": "FS7 + Type-2 weight", **internal_validity(X_sel, labels_a7)})

    # A8: rata-rata atas 100 subset acak berisi 7 fitur.
    nonconstant = [c for c in ALL_CRITERIA if c != "C10"]
    subsets = random_subsets(nonconstant, subset_size=n_features, n_subsets=args.n_random_subsets)
    sc_list, dbi_list, chi_list = [], [], []
    for subset in subsets:
        X_r = df_scaled[subset].to_numpy()
        _, labels_r = fit_kmeans(X_r, k)
        m = internal_validity(X_r, labels_r)
        sc_list.append(m["SC"]); dbi_list.append(m["DBI"]); chi_list.append(m["CHI"])
    rows.append({"variant": "A8", "setting": "Random FS7 mean", "SC": np.mean(sc_list), "DBI": np.mean(dbi_list), "CHI": np.mean(chi_list)})

    # A9: sequential backward selection yang murni data-driven.
    # C10 konstan (varians nol) sehingga dikeluarkan dari SBS, sama seperti pada subset acak A8
    # (naskah Bagian 3.2 dan 3.10: SBS memakai 15 indikator nonkonstan).
    X_nonconst = df_scaled[nonconstant].to_numpy()
    sbs_features = sequential_backward_selection(X_nonconst, nonconstant, k, floor_n_features=n_features)
    X_sbs = df_scaled[sbs_features].to_numpy()
    _, labels_sbs = fit_kmeans(X_sbs, k)
    rows.append({"variant": "A9", "setting": "SBS FS7", **internal_validity(X_sbs, labels_sbs)})

    table9 = pd.DataFrame(rows)
    table9.to_csv(outdir / "table_10_ablation.csv", index=False)
    print(table9.to_string(index=False))


if __name__ == "__main__":
    main()
