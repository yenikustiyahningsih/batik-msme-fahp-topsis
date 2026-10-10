"""Tahap 12: rata-rata waktu komputasi atas 30 kali fit berulang -> Tabel 14.

Nilai runtime bersifat spesifik-lingkungan dan hanya dipakai untuk perbandingan relatif
(manuskrip Bagian 4.9), bukan sebagai klaim performa absolut.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import clara, fcm, it2fcm, load_baseline_params, weighted_kmeans
from batik_fahp_topsis.clustering import fit_kmeans


def _time_it(fn, n_repeats: int = 30) -> tuple[float, float]:
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return float(np.mean(durations)), float(np.std(durations, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--fahp-weights", default="outputs/table_03_fahp_weights.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-repeats", type=int, default=30)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    features, k = config["features"], config["k"]
    X = df_scaled[features].to_numpy()

    weights_table = load_data(args.fahp_weights).set_index("criterion")
    type1_w = weights_table.loc[features, "type1_fahp_weight"].to_numpy()
    type1_w = type1_w / type1_w.sum()
    type2_w = weights_table.loc[features, "type2_fahp_weight"].to_numpy()
    type2_w = type2_w / type2_w.sum()

    params = load_baseline_params()
    rows = []

    mean_s, std_s = _time_it(lambda: fit_kmeans(X, k), args.n_repeats)
    rows.append({"method": "Proposed", "mean_s": mean_s, "SD": std_s})

    mean_s, std_s = _time_it(lambda: clara(X, k, params), args.n_repeats)
    rows.append({"method": "CLARA", "mean_s": mean_s, "SD": std_s})

    mean_s, std_s = _time_it(lambda: fcm(X, k, params), args.n_repeats)
    rows.append({"method": "FCM", "mean_s": mean_s, "SD": std_s})

    mean_s, std_s = _time_it(lambda: it2fcm(X, k, params), args.n_repeats)
    rows.append({"method": "IT2FCM", "mean_s": mean_s, "SD": std_s})

    mean_s, std_s = _time_it(lambda: weighted_kmeans(X, type1_w, k, params), args.n_repeats)
    rows.append({"method": "W-KM FAHP", "mean_s": mean_s, "SD": std_s})

    mean_s, std_s = _time_it(lambda: weighted_kmeans(X, type2_w, k, params), args.n_repeats)
    rows.append({"method": "W-KM T2", "mean_s": mean_s, "SD": std_s})

    table14 = pd.DataFrame(rows)
    table14.to_csv(outdir / "table_15_runtime.csv", index=False)
    print(table14.to_string(index=False))


if __name__ == "__main__":
    main()
