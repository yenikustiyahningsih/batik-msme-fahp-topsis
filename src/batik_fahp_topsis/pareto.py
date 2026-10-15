
"""Pareto efficiency and mean-rank-based configuration selection.

Manuscript: Section 3.4, Algorithm 1, Step 3.

Mean ranks are computed over the complete Pareto-efficient population.
`leading_subset` is only for display and must not be used as the ranking
population.

Selection hierarchy:
1. Pareto-efficient candidates.
2. Lowest mean rank.
3. Bootstrap ARI >= 0.90.
4. Every cluster must contain at least `min_cluster_fraction` of samples.

If no candidate satisfies all selection constraints, raise an error instead
of silently returning an ineligible configuration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = ("SC", "DBI", "CHI")
CONFIG_COLUMNS = ("n_features", "k")


def _require_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def _validate_metrics(df: pd.DataFrame) -> None:
    """Validate clustering metrics before Pareto analysis."""
    _require_columns(df, METRIC_COLUMNS, "grid_results")

    if df.empty:
        raise ValueError("grid_results must not be empty.")

    metrics = df.loc[:, list(METRIC_COLUMNS)].to_numpy(dtype=float)

    if not np.isfinite(metrics).all():
        raise ValueError(
            "SC, DBI, and CHI must contain only finite values."
        )

    if (df["DBI"] < 0).any():
        raise ValueError("DBI must not contain negative values.")


def is_dominated(row: pd.Series, others: pd.DataFrame) -> bool:
    """Return whether another configuration dominates this configuration.

    SC and CHI: higher is better.
    DBI: lower is better.
    """
    _require_columns(others, METRIC_COLUMNS, "others")

    if others.empty:
        return False

    no_worse = (
        (others["SC"] >= row["SC"])
        & (others["DBI"] <= row["DBI"])
        & (others["CHI"] >= row["CHI"])
    )

    strictly_better = (
        (others["SC"] > row["SC"])
        | (others["DBI"] < row["DBI"])
        | (others["CHI"] > row["CHI"])
    )

    return bool((no_worse & strictly_better).any())


def pareto_efficient_set(grid_results: pd.DataFrame) -> pd.DataFrame:
    """Return all configurations with their Pareto-efficiency flags."""
    if not isinstance(grid_results, pd.DataFrame):
        raise TypeError("grid_results must be a pandas DataFrame.")

    _validate_metrics(grid_results)

    df = grid_results.copy().reset_index(drop=True)

    flags = []
    for i, row in df.iterrows():
        others = df.drop(index=i)
        flags.append(not is_dominated(row, others))

    df["pareto_efficient"] = flags
    return df


def compute_mean_ranks(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weight mean ranks within the full Pareto population.

    Tied metric values receive the minimum rank in their tie group.
    """
    if not isinstance(pareto_df, pd.DataFrame):
        raise TypeError("pareto_df must be a pandas DataFrame.")

    _validate_metrics(pareto_df)
    _require_columns(pareto_df, ("pareto_efficient",), "pareto_df")

    if pareto_df["pareto_efficient"].isna().any():
        raise ValueError("pareto_efficient must not contain missing values.")

    if not pareto_df["pareto_efficient"].isin([True, False]).all():
        raise ValueError("pareto_efficient must contain boolean values.")

    df = pareto_df.loc[pareto_df["pareto_efficient"]].copy()

    if df.empty:
        raise ValueError("No Pareto-efficient configurations were found.")

    df["rank_sc"] = df["SC"].rank(
        ascending=False, method="min"
    )
    df["rank_dbi"] = df["DBI"].rank(
        ascending=True, method="min"
    )
    df["rank_chi"] = df["CHI"].rank(
        ascending=False, method="min"
    )

    df["mean_rank"] = (
        df["rank_sc"] + df["rank_dbi"] + df["rank_chi"]
    ) / 3.0

    return df.sort_values(
        ["mean_rank", "n_features", "k"]
        if all(c in df.columns for c in CONFIG_COLUMNS)
        else ["mean_rank"],
        kind="stable",
    ).reset_index(drop=True)


def leading_subset(
    ranked_df: pd.DataFrame,
    n: int = 8,
) -> pd.DataFrame:
    """Return the top n rows for display only; do not rerank this subset."""
    if not isinstance(ranked_df, pd.DataFrame):
        raise TypeError("ranked_df must be a pandas DataFrame.")

    if "mean_rank" not in ranked_df.columns:
        raise ValueError("ranked_df must contain a mean_rank column.")

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    return ranked_df.sort_values(
        "mean_rank", kind="stable"
    ).head(n).copy()


def select_final_configuration(
    ranked_df: pd.DataFrame,
    bootstrap_ari: dict[tuple[int, int], float],
    min_cluster_fraction: float = 0.03,
    cluster_sizes: dict[tuple[int, int], list[int]] | None = None,
) -> pd.Series:
    """Select a configuration using the manuscript's stated hierarchy.

    `cluster_sizes` maps (n_features, k) to the cluster counts for that
    configuration. It is required so the minimum cluster-size constraint
    cannot be silently skipped.
    """
    if not isinstance(ranked_df, pd.DataFrame):
        raise TypeError("ranked_df must be a pandas DataFrame.")

    _require_columns(
        ranked_df,
        ("mean_rank", *CONFIG_COLUMNS),
        "ranked_df",
    )

    if ranked_df.empty:
        raise ValueError("ranked_df must not be empty.")

    if cluster_sizes is None:
        raise ValueError(
            "cluster_sizes is required to enforce the minimum cluster-size "
            "constraint. Update run_bootstrap.py to pass cluster counts for "
            "each (n_features, k) configuration."
        )

    if (
        not np.isfinite(min_cluster_fraction)
        or not 0 < min_cluster_fraction <= 1
    ):
        raise ValueError("min_cluster_fraction must be in (0, 1].")

    df = ranked_df.copy()

    for column in ("mean_rank", "n_features", "k"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(
        df[["mean_rank", "n_features", "k"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "mean_rank, n_features, and k must be finite."
        )

    if (df["n_features"] < 1).any() or (df["k"] < 2).any():
        raise ValueError("n_features must be >= 1 and k must be >= 2.")

    if not np.equal(df["n_features"], np.floor(df["n_features"])).all():
        raise ValueError("n_features must contain integers.")

    if not np.equal(df["k"], np.floor(df["k"])).all():
        raise ValueError("k must contain integers.")

    if df.duplicated(list(CONFIG_COLUMNS)).any():
        raise ValueError(
            "ranked_df contains duplicate (n_features, k) configurations."
        )

    if "pareto_efficient" in df.columns:
        df = df.loc[df["pareto_efficient"].eq(True)].copy()

    if df.empty:
        raise ValueError("No Pareto-efficient candidates are available.")

    eligible = []

    # Mean rank is the primary ordering. Bootstrap ARI is a tie-breaker
    # only after candidates pass the ARI threshold and size constraint.
    df = df.sort_values(
        ["mean_rank", "n_features", "k"],
        kind="stable",
    )

    for _, row in df.iterrows():
        key = (int(row["n_features"]), int(row["k"]))

        if key not in bootstrap_ari:
            continue

        ari = bootstrap_ari[key]
        if ari is None or not np.isfinite(ari) or ari < 0.90:
            continue

        if key not in cluster_sizes:
            continue

        sizes = cluster_sizes[key]
        if not sizes:
            continue

        sizes_array = np.asarray(sizes, dtype=float)

        if (
            sizes_array.ndim != 1
            or not np.isfinite(sizes_array).all()
            or (sizes_array <= 0).any()
            or len(sizes_array) != key[1]
        ):
            continue

        total = sizes_array.sum()
        if total <= 0:
            continue

        if (sizes_array / total < min_cluster_fraction).any():
            continue

        eligible.append((row, float(ari)))

    if not eligible:
        raise ValueError(
            "No configuration satisfies all selection constraints: "
            "Pareto-efficient, bootstrap ARI >= 0.90, and every cluster "
            f">= {min_cluster_fraction:.1%} of the sample. "
            "Inspect the candidate metrics and cluster-size distributions; "
            "do not silently fall back to an ineligible configuration."
        )

    # Mean rank is already the primary order; use ARI only to break ties.
    eligible.sort(
        key=lambda item: (
            float(item[0]["mean_rank"]),
            -item[1],
            int(item[0]["n_features"]),
            int(item[0]["k"]),
        )
    )

    return eligible[0][0].copy()
