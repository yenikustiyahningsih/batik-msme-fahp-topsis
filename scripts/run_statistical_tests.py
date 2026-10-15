
"""Stage 7: Friedman and Holm-corrected Wilcoxon tests.

Evaluates paired bootstrap results using the selected feature representation
and cluster count.

Type-1 FAHP is excluded when its genuine weights are unavailable.
No Type-2 substitution is performed.

Outputs:
- table_09_statistical_tests.csv
- table_09_holm_wilcoxon_SC.csv
- table_09_holm_wilcoxon_DBI.csv
- table_09_holm_wilcoxon_CHI.csv
- table_09_bootstrap_metric_values.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import (
    clara,
    fcm,
    it2fcm,
    load_baseline_params,
    weighted_kmeans,
)
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.reproducibility import bootstrap_seed
from batik_fahp_topsis.statistical_tests import (
    friedman_test,
    holm_wilcoxon,
)
from batik_fahp_topsis.validity_metrics import internal_validity


METRICS = ("SC", "DBI", "CHI")


def load_weights(
    table: pd.DataFrame,
    features: list[str],
    column: str,
    required: bool = False,
) -> np.ndarray | None:
    """Load and normalize weights without manufacturing missing values."""
    if "criterion" not in table.columns:
        raise ValueError("Weights table must contain 'criterion'.")

    if table["criterion"].duplicated().any():
        raise ValueError("Weights table contains duplicate criteria.")

    if column not in table.columns:
        if required:
            raise ValueError(f"Required weight column missing: {column}")
        return None

    indexed = table.set_index("criterion")

    missing = [feature for feature in features if feature not in indexed.index]
    if missing:
        raise ValueError(f"Missing criteria in weights table: {missing}")

    values = pd.to_numeric(
        indexed.loc[features, column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        if required:
            raise ValueError(f"Required weights are invalid: {column}")
        return None

    if (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"Invalid weight values: {column}")

    return values / values.sum()


def _bootstrap_metric_matrix(
    X: np.ndarray,
    k: int,
    params: dict,
    weights_t1: np.ndarray | None,
    weights_t2: np.ndarray,
    n_bootstrap: int = 30,
) -> dict[str, pd.DataFrame]:
    """Generate paired bootstrap metric matrices for available methods."""
    methods = [
        "Proposed",
        "CLARA",
        "FCM",
        "IT2FCM",
    ]

    if weights_t1 is not None:
        methods.append("Weighted K-Means (FAHP)")

    methods.append("Weighted K-Means (Type-2)")

    scores = {
        method: {metric: [] for metric in METRICS}
        for method in methods
    }

    n = X.shape[0]

    for b in range(n_bootstrap):
        seed = bootstrap_seed(b)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=n)
        Xb = X[indices]

        results = {}

        _, labels = fit_kmeans(
            Xb,
            k,
            seed=seed,
            n_init=params["global"]["n_init_bootstrap"],
        )
        results["Proposed"] = labels

        labels_clara, _ = clara(Xb, k, params)
        results["CLARA"] = labels_clara

        fcm_result = fcm(Xb, k, params)
        results["FCM"] = fcm_result["labels"]

        it2_result = it2fcm(Xb, k, params)
        results["IT2FCM"] = it2_result["labels"]

        if weights_t1 is not None:
            labels_wk1, _ = weighted_kmeans(
                Xb, weights_t1, k, params
            )
            results["Weighted K-Means (FAHP)"] = labels_wk1

        labels_wk2, _ = weighted_kmeans(
            Xb, weights_t2, k, params
        )
        results["Weighted K-Means (Type-2)"] = labels_wk2

        for method in methods:
            metrics = internal_validity(Xb, results[method])

            for metric in METRICS:
                value = float(metrics[metric])

                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite {metric} for {method}, "
                        f"bootstrap repetition {b + 1}."
                    )

                scores[method][metric].append(value)

    return {
        metric: pd.DataFrame(
            {method: scores[method][metric] for method in methods}
        )
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--n-bootstrap", type=int, default=30)
    args = parser.parse_args()

    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap must be at least 2.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        Path(args.selected_config).read_text(encoding="utf-8")
    )

    features = config["features"]
    k = int(config["k"])

    if not features or len(features) != len(set(features)):
        raise ValueError("Selected features are empty or duplicated.")

    if k < 2:
        raise ValueError("K must be at least 2.")

    df_scaled = load_data(args.scaled_data)

    missing = [
        feature for feature in features if feature not in df_scaled.columns
    ]
    if missing:
        raise ValueError(f"Missing selected features: {missing}")

    X = df_scaled[features].to_numpy(dtype=float)

    if X.ndim != 2 or len(X) <= k:
        raise ValueError("Dataset must contain more than K samples.")

    if not np.isfinite(X).all():
        raise ValueError("Selected feature matrix contains NaN or infinity.")

    weights_table = load_data(args.fahp_weights)

    type1_w = load_weights(
        weights_table,
        features,
        "type1_fahp_weight",
        required=False,
    )

    type2_w = load_weights(
        weights_table,
        features,
        "type2_fahp_weight",
        required=True,
    )

    if type1_w is None:
        print(
            "WARNING: Type-1 FAHP weights unavailable. "
            "Weighted K-Means (FAHP) is excluded from statistical tests."
        )

    params = load_baseline_params()

    paired = _bootstrap_metric_matrix(
        X,
        k,
        params,
        type1_w,
        type2_w,
        n_bootstrap=args.n_bootstrap,
    )

    # Save all observed bootstrap metric values for auditing.
    audit_rows = []

    for metric, matrix in paired.items():
        for repetition, (_, values) in enumerate(
            matrix.iterrows(), start=1
        ):
            for method, value in values.items():
                audit_rows.append(
                    {
                        "metric": metric,
                        "repetition": repetition,
                        "method": method,
                        "value": float(value),
                    }
                )

    pd.DataFrame(audit_rows).to_csv(
        outdir / "table_09_bootstrap_metric_values.csv",
        index=False,
    )

    summary_rows = []

    for metric, paired_df in paired.items():
        if paired_df.isna().any().any():
            raise ValueError(
                f"Missing values in paired bootstrap matrix: {metric}"
            )

        friedman = friedman_test(paired_df)

        holm = holm_wilcoxon(
            paired_df,
            reference_col="Proposed",
        )

        n_significant = int(
            holm["significant_at_alpha"].sum()
        )

        summary_rows.append(
            {
                "metric": metric,
                "chi_square": round(
                    float(friedman["chi_square"]), 2
                ),
                "p_value": float(friedman["p_value"]),
                "holm_significant_pairs": (
                    f"{n_significant}/{len(holm)}"
                ),
                "n_methods": len(paired_df.columns),
                "n_bootstrap": len(paired_df),
                "type1_fahp_available": type1_w is not None,
            }
        )

        holm.to_csv(
            outdir / f"table_09_holm_wilcoxon_{metric}.csv",
            index=False,
        )

    summary = pd.DataFrame(summary_rows)

    output_path = outdir / "table_09_statistical_tests.csv"
    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nStatistical tests saved to: {output_path}")
    print(
        "Note: bootstrap replicates share the same resampled indices "
        "across methods. Because replicates overlap, the resulting "
        "p-values should be interpreted cautiously rather than as "
        "independent external validation."
    )


if __name__ == "__main__":
    main()
