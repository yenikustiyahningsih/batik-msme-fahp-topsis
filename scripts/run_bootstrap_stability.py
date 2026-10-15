
"""Stage 5: bootstrap stability for Pareto-efficient configurations.

Outputs:
- table_07_pareto_bootstrap.csv
- table_07_bootstrap_per_repetition.csv
- table_12_cluster_distribution.csv
- selected_configuration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.bootstrap import (
    bootstrap_stability,
    cluster_size_distribution,
)
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.pareto import select_final_configuration


def parse_ranked_criteria(path: str) -> list[str]:
    """Load ordered criteria from a comma-separated text file."""
    criteria_path = Path(path)

    if not criteria_path.is_file():
        raise FileNotFoundError(
            f"Ranked-criteria file not found: {criteria_path}"
        )

    criteria = [
        item.strip()
        for item in criteria_path.read_text(encoding="utf-8").split(",")
        if item.strip()
    ]

    if not criteria:
        raise ValueError("The ranked-criteria file contains no criteria.")

    if len(criteria) != len(set(criteria)):
        raise ValueError("The ranked-criteria file contains duplicates.")

    return criteria


def validate_inputs(
    df_scaled: pd.DataFrame,
    ranked: pd.DataFrame,
    ranked_criteria: list[str],
    n_bootstrap: int,
) -> None:
    """Validate dataset, candidate configurations, and bootstrap settings."""
    if n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")

    if df_scaled.empty:
        raise ValueError("The scaled dataset is empty.")

    if "msme_id" not in df_scaled.columns:
        raise ValueError("The scaled dataset must contain 'msme_id'.")

    if df_scaled["msme_id"].isna().any():
        raise ValueError("'msme_id' contains missing values.")

    if df_scaled["msme_id"].duplicated().any():
        raise ValueError("'msme_id' must be unique.")

    if ranked.empty:
        raise ValueError("The ranked configuration table is empty.")

    required = {"n_features", "k", "mean_rank", *("SC", "DBI", "CHI")}
    missing = sorted(required - set(ranked.columns))

    if missing:
        raise ValueError(
            f"Ranked table is missing required columns: {missing}. "
            "Run the Pareto/ranking step first."
        )

    if "pareto_efficient" not in ranked.columns:
        raise ValueError(
            "Ranked table must contain 'pareto_efficient'. "
            "Run the Pareto/ranking step first."
        )

    if ranked["pareto_efficient"].isna().any():
        raise ValueError("'pareto_efficient' contains missing values.")

    if not ranked["pareto_efficient"].isin([True, False]).all():
        raise ValueError("'pareto_efficient' must contain boolean values.")

    if not ranked["pareto_efficient"].any():
        raise ValueError("No Pareto-efficient configurations were found.")

    for column in ("n_features", "k", "mean_rank", "SC", "DBI", "CHI"):
        ranked[column] = pd.to_numeric(ranked[column], errors="raise")

    numeric = ranked[
        ["n_features", "k", "mean_rank", "SC", "DBI", "CHI"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Configuration metrics contain non-finite values.")

    for column in ("n_features", "k"):
        if not np.equal(
            ranked[column], np.floor(ranked[column])
        ).all():
            raise ValueError(f"'{column}' must contain integers.")

    if (ranked["n_features"] < 1).any():
        raise ValueError("'n_features' must be at least 1.")

    if (ranked["k"] < 2).any():
        raise ValueError("'k' must be at least 2.")

    if ranked.duplicated(["n_features", "k"]).any():
        raise ValueError(
            "Duplicate (n_features, k) configurations found."
        )

    if len(ranked_criteria) < int(ranked["n_features"].max()):
        raise ValueError(
            "The ranked-criteria file contains fewer features than "
            "required by the candidate configurations."
        )

    missing_features = [
        feature
        for feature in ranked_criteria[: int(ranked["n_features"].max())]
        if feature not in df_scaled.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features missing from scaled dataset: {missing_features}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bootstrap stability and select a configuration."
    )
    parser.add_argument(
        "--scaled-data",
        default="outputs/preprocessed_scaled.csv",
    )
    parser.add_argument(
        "--ranked-table",
        default="outputs/table_07_pareto_bootstrap.csv",
    )
    parser.add_argument(
        "--ranked-criteria",
        default="outputs/fahp_ranked_criteria.txt",
    )
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-bootstrap", type=int, default=30)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    ranked = load_data(args.ranked_table)
    ranked_criteria = parse_ranked_criteria(args.ranked_criteria)

    validate_inputs(
        df_scaled,
        ranked,
        ranked_criteria,
        args.n_bootstrap,
    )

    # Only evaluate the complete Pareto-efficient population.
    candidates = ranked.loc[ranked["pareto_efficient"]].copy()

    bootstrap_ari: dict[tuple[int, int], float] = {}
    bootstrap_results: dict[tuple[int, int], dict] = {}
    cluster_sizes: dict[tuple[int, int], list[int]] = {}

    for _, row in candidates.iterrows():
        n_features = int(row["n_features"])
        k = int(row["k"])
        key = (n_features, k)

        features = ranked_criteria[:n_features]
        X = df_scaled[features].to_numpy(dtype=float)

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite values found for configuration {key}."
            )

        if k > len(X):
            raise ValueError(
                f"K={k} exceeds the sample count for configuration {key}."
            )

        _, reference_labels = fit_kmeans(X, k)

        stability = bootstrap_stability(
            X,
            k,
            reference_labels,
            n_bootstrap=args.n_bootstrap,
        )

        distribution = cluster_size_distribution(reference_labels)

        bootstrap_ari[key] = float(stability["ARI_mean"])
        bootstrap_results[key] = stability

        # Preserve one count per cluster label. Missing clusters will fail
        # the selection validation rather than being silently accepted.
        cluster_sizes[key] = [
            int(distribution.get(cluster, 0))
            for cluster in range(k)
        ]

    # Merge bootstrap metrics into the full input table, preserving rows.
    ranked["bootstrap_ARI_mean"] = np.nan
    ranked["bootstrap_NMI_mean"] = np.nan
    ranked["bootstrap_ARI_sd"] = np.nan
    ranked["bootstrap_NMI_sd"] = np.nan

    for index, row in ranked.iterrows():
        key = (int(row["n_features"]), int(row["k"]))
        result = bootstrap_results.get(key)

        if result is not None:
            ranked.at[index, "bootstrap_ARI_mean"] = result["ARI_mean"]
            ranked.at[index, "bootstrap_NMI_mean"] = result["NMI_mean"]
            ranked.at[index, "bootstrap_ARI_sd"] = result["ARI_std"]
            ranked.at[index, "bootstrap_NMI_sd"] = result["NMI_std"]

    ranked.to_csv(
        outdir / "table_07_pareto_bootstrap.csv",
        index=False,
    )

    # Save per-repetition values for reproducibility and SD auditing.
    per_rep = []

    for (n_features, k), result in bootstrap_results.items():
        for repetition, (ari, nmi) in enumerate(
            zip(
                result["ARI_per_repetition"],
                result["NMI_per_repetition"],
            ),
            start=1,
        ):
            per_rep.append(
                {
                    "n_features": n_features,
                    "k": k,
                    "repetition": repetition,
                    "ARI": ari,
                    "NMI": nmi,
                }
            )

    pd.DataFrame(
        per_rep,
        columns=[
            "n_features",
            "k",
            "repetition",
            "ARI",
            "NMI",
        ],
    ).to_csv(
        outdir / "table_07_bootstrap_per_repetition.csv",
        index=False,
    )

    # Select using mean rank, bootstrap ARI threshold, and cluster sizes.
    selected = select_final_configuration(
        candidates,
        bootstrap_ari,
        min_cluster_fraction=0.03,
        cluster_sizes=cluster_sizes,
    )

    n_features = int(selected["n_features"])
    k = int(selected["k"])
    features = ranked_criteria[:n_features]
    selected_key = (n_features, k)

    X = df_scaled[features].to_numpy(dtype=float)
    _, labels = fit_kmeans(X, k)

    distribution = cluster_size_distribution(labels)
    total = sum(distribution.values())

    if total != len(df_scaled):
        raise RuntimeError(
            "Cluster counts do not sum to the dataset sample count."
        )

    table_distribution = pd.DataFrame(
        [
            {
                "cluster": cluster,
                "n": count,
                "pct": round(100 * count / total, 2),
            }
            for cluster, count in sorted(distribution.items())
        ]
    )

    table_distribution.to_csv(
        outdir / "table_12_cluster_distribution.csv",
        index=False,
    )

    selected_result = bootstrap_results[selected_key]

    selected_json = {
        "n_features": n_features,
        "k": k,
        "features": features,
        "mean_rank": float(selected["mean_rank"]),
        "bootstrap_ARI_mean": float(selected_result["ARI_mean"]),
        "bootstrap_ARI_sd": float(selected_result["ARI_std"]),
        "bootstrap_NMI_mean": float(selected_result["NMI_mean"]),
        "bootstrap_NMI_sd": float(selected_result["NMI_std"]),
        "n_bootstrap": args.n_bootstrap,
        "cluster_sizes": {
            str(cluster): count
            for cluster, count in sorted(distribution.items())
        },
        "min_cluster_fraction": 0.03,
    }

    with open(
        outdir / "selected_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selected_json, file, indent=2, ensure_ascii=False)

    print(
        f"Selected configuration: N={n_features}, K={k}, "
        f"mean-rank={selected['mean_rank']:.4f}"
    )
    print(f"Cluster distribution: {distribution}")
    print(
        f"Bootstrap ARI: {selected_result['ARI_mean']:.4f} "
        f"(SD {selected_result['ARI_std']:.4f})"
    )
    print(
        f"Bootstrap NMI: {selected_result['NMI_mean']:.4f} "
        f"(SD {selected_result['NMI_std']:.4f})"
    )


if __name__ == "__main__":
    main()
