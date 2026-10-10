"""Tahap 5: stabilitas bootstrap untuk setiap konfigurasi Pareto-efficient -> Tabel 6 (kolom
ARI/NMI), Tabel 11 (distribusi cluster akhir untuk konfigurasi yang terpilih)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import load_data
from batik_fahp_topsis.bootstrap import bootstrap_stability, cluster_size_distribution
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.pareto import select_final_configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", default="outputs/preprocessed_scaled.csv")
    parser.add_argument("--ranked-table", default="outputs/table_07_pareto_bootstrap.csv")
    parser.add_argument("--ranked-criteria", default="outputs/fahp_ranked_criteria.txt")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-bootstrap", type=int, default=30)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    ranked = load_data(args.ranked_table)
    ranked_criteria = Path(args.ranked_criteria).read_text(encoding="utf-8").strip().split(",")

    bootstrap_ari = {}
    bootstrap_results = {}
    for _, row in ranked.iterrows():
        n_features, k = int(row["n_features"]), int(row["k"])
        features = ranked_criteria[:n_features]
        X = df_scaled[features].to_numpy()
        _, reference_labels = fit_kmeans(X, k)
        stability = bootstrap_stability(X, k, reference_labels, n_bootstrap=args.n_bootstrap)
        bootstrap_ari[(n_features, k)] = stability["ARI_mean"]
        bootstrap_results[(n_features, k)] = stability

    ranked["bootstrap_ARI_mean"] = ranked.apply(lambda r: bootstrap_ari[(int(r["n_features"]), int(r["k"]))], axis=1)
    ranked["bootstrap_NMI_mean"] = ranked.apply(
        lambda r: bootstrap_results[(int(r["n_features"]), int(r["k"]))]["NMI_mean"], axis=1
    )
    ranked["bootstrap_ARI_sd"] = ranked.apply(
        lambda r: bootstrap_results[(int(r["n_features"]), int(r["k"]))]["ARI_std"], axis=1
    )
    ranked["bootstrap_NMI_sd"] = ranked.apply(
        lambda r: bootstrap_results[(int(r["n_features"]), int(r["k"]))]["NMI_std"], axis=1
    )
    ranked.to_csv(outdir / "table_07_pareto_bootstrap.csv", index=False)

    # Nilai ARI/NMI per repetisi untuk setiap konfigurasi Pareto-efficient (audit SD bootstrap).
    per_rep = []
    for (n_f, k_), res in bootstrap_results.items():
        for b, (ari, nmi) in enumerate(zip(res["ARI_per_repetition"], res["NMI_per_repetition"])):
            per_rep.append({"n_features": n_f, "k": k_, "repetition": b + 1, "ARI": ari, "NMI": nmi})
    pd.DataFrame(per_rep).to_csv(outdir / "table_07_bootstrap_per_repetition.csv", index=False)

    selected = select_final_configuration(ranked, bootstrap_ari)
    n_features, k = int(selected["n_features"]), int(selected["k"])
    features = ranked_criteria[:n_features]
    X = df_scaled[features].to_numpy()
    _, labels = fit_kmeans(X, k)

    distribution = cluster_size_distribution(labels)
    total = sum(distribution.values())
    table11 = pd.DataFrame(
        [{"cluster": c, "n": n, "pct": round(100 * n / total, 2)} for c, n in sorted(distribution.items())]
    )
    table11.to_csv(outdir / "table_12_cluster_distribution.csv", index=False)

    with open(outdir / "selected_configuration.json", "w", encoding="utf-8") as f:
        sel = bootstrap_results[(n_features, k)]
        json.dump(
            {
                "n_features": n_features,
                "k": k,
                "features": features,
                "bootstrap_ARI_mean": sel["ARI_mean"],
                "bootstrap_ARI_sd": sel["ARI_std"],
                "bootstrap_NMI_mean": sel["NMI_mean"],
                "bootstrap_NMI_sd": sel["NMI_std"],
                "n_bootstrap": args.n_bootstrap,
            },
            f,
            indent=2,
        )

    print(f"Konfigurasi terpilih berdasarkan hierarki yang dinyatakan: N={n_features}, K={k}")
    print(f"Distribusi cluster: {distribution}")
    sel = bootstrap_results[(n_features, k)]
    print(f"Bootstrap konfigurasi terpilih: ARI {sel['ARI_mean']:.4f} (SD {sel['ARI_std']:.4f}), "
          f"NMI {sel['NMI_mean']:.4f} (SD {sel['NMI_std']:.4f})")


if __name__ == "__main__":
    main()
