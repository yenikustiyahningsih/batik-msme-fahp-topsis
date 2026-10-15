
"""Stage 6: baseline comparison using the selected feature representation and K.

The comparison evaluates alternative partitioning mechanisms on the selected
representation and number of clusters. It is not an independently tuned
benchmark.

If FAHP Type-1 weights are unavailable, the corresponding baseline is marked
as unavailable rather than substituting Type-2 weights.
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
from batik_fahp_topsis.validity_metrics import internal_validity


def load_normalized_weights(
    weights_table: pd.DataFrame,
    features: list[str],
    column: str,
) -> np.ndarray | None:
    """Return normalized weights, or None if valid weights are unavailable."""
    if column not in weights_table.columns:
        return None

    if "criterion" not in weights_table.columns:
        raise ValueError("FAHP weight table must contain 'criterion'.")

    indexed = weights_table.set_index("criterion")

    if indexed.index.duplicated().any():
        raise ValueError("FAHP weight table contains duplicate criteria.")

    missing = [feature for feature in features if feature not in indexed.index]
    if missing:
        raise ValueError(
            f"FAHP weight table is missing selected criteria: {missing}"
        )

    values = pd.to_numeric(
        indexed.loc[features, column],
        errors="coerce",
    ).to_numpy(dtype=float)

    # Missing Type-1 weights are an unavailable baseline, not zero weights.
    if not np.isfinite(values).all():
        return None

    if (values < 0).any() or values.sum() <= 0:
        raise ValueError(
            f"Invalid values in weight column '{column}'."
        )

    return values / values.sum()


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
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate and load selected configuration.
    config_path = Path(args.selected_config)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Selected configuration not found: {config_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if "features" not in config or "k" not in config:
        raise ValueError(
            "Selected configuration must contain 'features' and 'k'."
        )

    features = config["features"]
    k = int(config["k"])

    if not features or k < 2:
        raise ValueError("Selected features must be nonempty and K >= 2.")

    df_scaled = load_data(args.scaled_data)

    missing_features = [
        feature for feature in features if feature not in df_scaled.columns
    ]
    if missing_features:
        raise ValueError(
            f"Selected features are missing from scaled data: "
            f"{missing_features}"
        )

    X = df_scaled[features].to_numpy(dtype=float)

    if X.ndim != 2 or X.shape[0] < k or not np.isfinite(X).all():
        raise ValueError(
            "Selected feature data must be finite and contain at least K rows."
        )

    weights_path = Path(args.fahp_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"FAHP weights file not found: {weights_path}"
        )

    weights_table = load_data(str(weights_path))

    type2_w = load_normalized_weights(
        weights_table,
        features,
        "type2_fahp_weight",
    )
    type1_w = load_normalized_weights(
        weights_table,
        features,
        "type1_fahp_weight",
    )

    if type2_w is None:
        raise ValueError(
            "Valid Type-2 FAHP weights are required for this comparison."
        )

    params = load_baseline_params()
    rows = []

    def append_result(method: str, labels: np.ndarray) -> None:
        metrics = internal_validity(X, labels)
        rows.append({"method": method, **metrics, "status": "computed"})

    # Proposed method.
    _, labels_proposed = fit_kmeans(
        X,
        k,
        seed=params["global"]["random_state"],
    )
    append_result("Proposed", labels_proposed)

    # Alternative partitioning methods.
    labels_clara, _ = clara(X, k, params)
    append_result("CLARA", labels_clara)

    fcm_result = fcm(X, k, params)
    append_result("FCM", fcm_result["labels"])

    it2fcm_result = it2fcm(X, k, params)
    append_result("IT2FCM", it2fcm_result["labels"])

    # Type-1 baseline is only computed when its own weights are available.
    if type1_w is None:
        rows.append(
            {
                "method": "Weighted K-Means (FAHP)",
                "status": "not_evaluated",
                "note": (
                    "Type-1 FAHP weights unavailable; no Type-2 substitution."
                ),
            }
        )
        print(
            "WARNING: Weighted K-Means (FAHP) skipped because valid "
            "Type-1 weights are unavailable."
        )
    else:
        labels_wk_fahp, _ = weighted_kmeans(X, type1_w, k, params)
        append_result("Weighted K-Means (FAHP)", labels_wk_fahp)

    # Type-2 baseline.
    labels_wk_t2, _ = weighted_kmeans(X, type2_w, k, params)
    append_result("Weighted K-Means (Type-2)", labels_wk_t2)

    table = pd.DataFrame(rows)

    output_path = outdir / "table_08_baseline_comparison.csv"
    table.to_csv(output_path, index=False)

    print(table.to_string(index=False))
    print(f"\nBaseline comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
