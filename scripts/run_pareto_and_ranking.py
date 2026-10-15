
"""Stage 4: Pareto efficiency and mean-rank calculation.

Outputs:
- table_07_pareto_ranking.csv: complete Pareto-efficient population
- table_06_leading_configs.csv: leading configurations for display only

Bootstrap metrics are added later by run_bootstrap_stability.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_data
from batik_fahp_topsis.pareto import (
    compute_mean_ranks,
    leading_subset,
    pareto_efficient_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Pareto-efficient configurations and mean ranks."
    )
    parser.add_argument(
        "--results",
        default="outputs/results_90_configs.csv",
    )
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--leading-n", type=int, default=8)
    args = parser.parse_args()

    if args.leading_n < 1:
        parser.error("--leading-n must be at least 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = load_data(args.results)

    # Identify the complete Pareto-efficient population.
    pareto_df = pareto_efficient_set(grid)

    # Calculate ranks across the entire Pareto-efficient population.
    ranked = compute_mean_ranks(pareto_df)

    if ranked.empty:
        raise RuntimeError(
            "No Pareto-efficient configurations were produced."
        )

    # Save the complete ranking separately from the bootstrap output.
    ranking_path = outdir / "table_07_pareto_ranking.csv"
    ranked.to_csv(ranking_path, index=False)

    # This subset is for presentation only; never rerank this subset.
    leading = leading_subset(ranked, n=args.leading_n)
    leading_path = outdir / "table_06_leading_configs.csv"
    leading.to_csv(leading_path, index=False)

    best = ranked.iloc[0]

    print(f"Total grid configurations: {len(grid)}")
    print(f"Pareto-efficient configurations: {len(ranked)}")
    print(
        "Lowest mean-rank configuration: "
        f"N={int(best['n_features'])}, "
        f"K={int(best['k'])}, "
        f"mean_rank={best['mean_rank']:.4f}"
    )
    print(f"Complete ranking saved to: {ranking_path}")
    print(f"Leading subset saved to: {leading_path}")
    print(
        "Note: the lowest mean-rank configuration is not automatically "
        "the final selection. Run bootstrap stability and apply the "
        "declared stability and cluster-size constraints."
    )


if __name__ == "__main__":
    main()
