"""Tahap 7: Friedman + Wilcoxon yang dikoreksi Holm atas 30 pasangan hasil bootstrap -> Tabel 8."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import clara, fcm, it2fcm, load_baseline_params, weighted_kmeans
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.reproducibility import bootstrap_seed
from batik_fahp_topsis.statistical_tests import friedman_test, holm_wilcoxon
from batik_fahp_topsis.validity_metrics import internal_validity


def _bootstrap_metric_matrix(X: np.ndarray, k: int, params: dict, weights_t1, weights_t2, n_bootstrap: int = 30):
    methods = ["Proposed", "CLARA", "FCM", "IT2FCM", "Weighted K-Means (FAHP)", "Weighted K-Means (Type-2)"]
    scores = {m: {"SC": [], "DBI": [], "CHI": []} for m in methods}

    n = X.shape[0]
    for b in range(n_bootstrap):
        rng = np.random.default_rng(bootstrap_seed(b))
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]

        _, labels = fit_kmeans(Xb, k, seed=bootstrap_seed(b), n_init=params["global"]["n_init_bootstrap"])
        m = internal_validity(Xb, labels)
        for key in ("SC", "DBI", "CHI"):
            scores["Proposed"][key].append(m[key])

        labels_clara, _ = clara(Xb, k, params)
        m = internal_validity(Xb, labels_clara)
        for key in ("SC", "DBI", "CHI"):
            scores["CLARA"][key].append(m[key])

        fcm_r = fcm(Xb, k, params)
        m = internal_validity(Xb, fcm_r["labels"])
        for key in ("SC", "DBI", "CHI"):
            scores["FCM"][key].append(m[key])

        it2_r = it2fcm(Xb, k, params)
        m = internal_validity(Xb, it2_r["labels"])
        for key in ("SC", "DBI", "CHI"):
            scores["IT2FCM"][key].append(m[key])

        labels_wk1, _ = weighted_kmeans(Xb, weights_t1, k, params)
        m = internal_validity(Xb, labels_wk1)
        for key in ("SC", "DBI", "CHI"):
            scores["Weighted K-Means (FAHP)"][key].append(m[key])

        labels_wk2, _ = weighted_kmeans(Xb, weights_t2, k, params)
        m = internal_validity(Xb, labels_wk2)
        for key in ("SC", "DBI", "CHI"):
            scores["Weighted K-Means (Type-2)"][key].append(m[key])

    return {metric: pd.DataFrame({m: scores[m][metric] for m in methods}) for metric in ("SC", "DBI", "CHI")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--selected-config", default="outputs/selected_configuration.json")
    parser.add_argument("--fahp-weights", default="outputs/table_03_fahp_weights.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-bootstrap", type=int, default=30)
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
    paired = _bootstrap_metric_matrix(X, k, params, type1_w, type2_w, n_bootstrap=args.n_bootstrap)

    rows = []
    for metric, paired_df in paired.items():
        friedman = friedman_test(paired_df)
        holm = holm_wilcoxon(paired_df, reference_col="Proposed")
        n_sig = int(holm["significant_at_alpha"].sum())
        rows.append(
            {
                "metric": metric,
                "chi_square": round(friedman["chi_square"], 2),
                "p_value": friedman["p_value"],
                "holm_significant_pairs": f"{n_sig}/{len(holm)}",
            }
        )
        holm.to_csv(outdir / f"table_09_holm_wilcoxon_{metric}.csv", index=False)

    table8 = pd.DataFrame(rows)
    table8.to_csv(outdir / "table_09_statistical_tests.csv", index=False)
    print(table8.to_string(index=False))


if __name__ == "__main__":
    main()
