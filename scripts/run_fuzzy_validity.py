"""Tahap 9: validitas khusus-fuzzy (PC, PE, XB, FOU) untuk FCM / IT2FCM -> Tabel 10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import fcm, it2fcm, load_baseline_params
from batik_fahp_topsis.fahp import footprint_of_uncertainty
from batik_fahp_topsis.validity_metrics import partition_coefficient, partition_entropy, xie_beni_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = config["features"], config["k"]
    X = df_scaled[features].to_numpy()

    params = load_baseline_params()
    rows = []

    fcm_r = fcm(X, k, params)
    pc = partition_coefficient(fcm_r["membership"])
    pe = partition_entropy(fcm_r["membership"])
    xb = xie_beni_index(X, fcm_r["membership"], fcm_r["centers"], m=params["fcm"]["fuzzy_exponent_m"])
    rows.append({"model": "FCM", "PC": pc, "PE": pe, "XB": xb, "FOU": None})

    it2_r = it2fcm(X, k, params)
    pc2 = partition_coefficient(it2_r["membership"])
    pe2 = partition_entropy(it2_r["membership"])
    xb2 = xie_beni_index(X, it2_r["membership"], it2_r["centers"], m=params["it2fcm"]["fuzzy_exponent_m_upper"])
    fou = footprint_of_uncertainty(it2_r["umf"], it2_r["lmf"])
    rows.append({"model": "IT2FCM", "PC": pc2, "PE": pe2, "XB": xb2, "FOU": fou})

    table10 = pd.DataFrame(rows)
    table10.to_csv(outdir / "table_11_fuzzy_validity.csv", index=False)
    print(table10.to_string(index=False))


if __name__ == "__main__":
    main()
