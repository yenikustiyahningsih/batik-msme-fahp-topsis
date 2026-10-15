
"""Main pipeline for generating manuscript tables and figures.

Runs preprocessing, FAHP weighting, joint optimization, Pareto ranking,
bootstrap stability, evaluation, TOPSIS, runtime benchmarking, and figures.

Usage:
    python scripts/generate_all_tables.py \
        --data data/example_data.csv \
        --outdir outputs \
        --n-bootstrap 30
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(script_name: str, argv: list[str]) -> None:
    """Run a pipeline script with isolated command-line arguments."""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required pipeline script was not found: {script_path}"
        )

    print(f"\n=== Running {script_name} {' '.join(argv)} ===")

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def _require_file(path: str | Path, description: str) -> None:
    """Fail early if a required input file does not exist."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {file_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        default="data/example_data.csv",
        help="Input dataset CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap repetitions.",
    )
    parser.add_argument(
        "--imputed",
        default=None,
        help="Frozen imputed CSV, if applicable.",
    )
    parser.add_argument(
        "--weights-csv",
        default=None,
        help=(
            "CSV containing existing FAHP Type-2 weights. "
            "If omitted, run_fahp_weights.py uses its configured default."
        ),
    )

    args = parser.parse_args()

    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap must be at least 2.")

    _require_file(args.data, "Input dataset")

    if args.imputed is not None:
        _require_file(args.imputed, "Imputed dataset")

    if args.weights_csv is not None:
        _require_file(args.weights_csv, "FAHP weights CSV")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = str(outdir)
    common = ["--outdir", out]

    # Stage 1: preprocessing.
    preprocessing_args = [
        "--data", args.data,
        *common,
    ]
    if args.imputed is not None:
        preprocessing_args.extend(["--imputed", args.imputed])

    _run("run_preprocessing.py", preprocessing_args)

    scaled_data = str(outdir / "preprocessed_scaled.csv")
    ranked_criteria = str(outdir / "fahp_ranked_criteria.txt")
    weights_table = str(outdir / "table_03_fahp_weights.csv")

    _require_file(scaled_data, "Scaled preprocessing output")

    # Stage 2: FAHP weights.
    weight_args = list(common)
    if args.weights_csv is not None:
        weight_args.extend(["--weights-csv", args.weights_csv])

    _run("run_fahp_weights.py", weight_args)

    _require_file(ranked_criteria, "Ranked-criteria output")
    _require_file(weights_table, "FAHP weights output")

    # Stage 3: joint optimization.
    _run(
        "run_joint_optimization.py",
        [
            "--scaled-data", scaled_data,
            "--ranked-criteria", ranked_criteria,
            *common,
        ],
    )

    results_90 = str(outdir / "results_90_configs.csv")
    _require_file(results_90, "90-configuration optimization results")

    # Stage 4: Pareto efficiency and mean ranking.
    _run(
        "run_pareto_and_ranking.py",
        [
            "--results", results_90,
            *common,
        ],
    )

    # The ranking-only file is deliberately separate from the file that
    # receives bootstrap metrics.
    ranked_table = str(outdir / "table_07_pareto_ranking.csv")
    _require_file(ranked_table, "Pareto ranking output")

    # Stage 5: bootstrap stability and final configuration selection.
    _run(
        "run_bootstrap_stability.py",
        [
            "--scaled-data", scaled_data,
            "--ranked-table", ranked_table,
            "--ranked-criteria", ranked_criteria,
            "--n-bootstrap", str(args.n_bootstrap),
            *common,
        ],
    )

    selected_config = str(outdir / "selected_configuration.json")
    _require_file(selected_config, "Selected configuration")

    # Shared arguments for downstream analyses.
    shared_cfg = [
        "--scaled-data", scaled_data,
        "--selected-config", selected_config,
        "--fahp-weights", weights_table,
        *common,
    ]

    # Stage 6: evaluation and ablation.
    _run("run_baseline_comparison.py", shared_cfg)

    _run(
        "run_statistical_tests.py",
        [
            "--n-bootstrap", str(args.n_bootstrap),
            *shared_cfg,
        ],
    )

    _run("run_ablation.py", shared_cfg)

    _run(
        "run_fuzzy_validity.py",
        [
            "--scaled-data", scaled_data,
            "--selected-config", selected_config,
            *common,
        ],
    )

    # Stage 7: TOPSIS priorities and sensitivity.
    # Use the imputed dataset only when explicitly provided.
    topsis_extra = (
        ["--topsis-data", args.imputed]
        if args.imputed is not None
        else []
    )

    _run(
        "run_topsis_priorities.py",
        [*shared_cfg, *topsis_extra],
    )

    _run(
        "run_topsis_sensitivity.py",
        [*shared_cfg, *topsis_extra],
    )

    # Stage 8: runtime benchmark.
    _run("run_runtime_benchmark.py", shared_cfg)

    # Stage 9: cluster visualizations.
    _run(
        "run_cluster_figures.py",
        [
            "--scaled-data", scaled_data,
            "--results", results_90,
            "--selected-config", selected_config,
            *common,
        ],
    )

    print(
        f"\nPipeline completed. Outputs were written to '{outdir}'."
    )
    print(
        "Important: completion confirms that scripts ran successfully; "
        "it does not independently validate scientific conclusions or "
        "guarantee that every manuscript table has been generated."
    )


if __name__ == "__main__":
    main()
