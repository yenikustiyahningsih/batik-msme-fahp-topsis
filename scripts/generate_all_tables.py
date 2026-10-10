"""Script pipeline utama (Reviewer 1, Komentar 7): menjalankan setiap tahap secara berurutan
dari file data masukan yang sama, sehingga status Pareto, rank, konfigurasi terpilih, statistik
bootstrap, dan setiap tabel manuskrip dihasilkan secara otomatis dari satu file hasil otoritatif,
bukan ditranskripsi manual antar tahap.

Cara pakai:
    python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs/
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(script_name: str, argv: list[str]):
    print(f"\n=== Menjalankan {script_name} {' '.join(argv)} ===")
    old_argv = sys.argv
    sys.argv = [script_name] + argv
    try:
        runpy.run_path(str(SCRIPTS_DIR / script_name), run_name="__main__")
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/example_data.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--imputed", default=None, help="CSV imputasi beku (lihat run_preprocessing.py).")
    parser.add_argument(
        "--weights-csv",
        default=None,
        help="Pakai bobot FAHP Type-2 yang sudah ada (mis. params/paper_table3_weights.csv) "
        "alih-alih matriks pakar sintetis.",
    )
    args = parser.parse_args()

    outdir = args.outdir
    common = ["--outdir", outdir]

    _run("run_preprocessing.py", ["--data", args.data, *common] + (["--imputed", args.imputed] if args.imputed else []))
    _run("run_fahp_weights.py", common + (["--weights-csv", args.weights_csv] if args.weights_csv else []))
    _run(
        "run_joint_optimization.py",
        [
            "--scaled-data", f"{outdir}/preprocessed_scaled.csv",
            "--ranked-criteria", f"{outdir}/fahp_ranked_criteria.txt",
            *common,
        ],
    )
    _run("run_pareto_and_ranking.py", ["--results", f"{outdir}/results_90_configs.csv", *common])
    _run(
        "run_bootstrap_stability.py",
        [
            "--scaled-data", f"{outdir}/preprocessed_scaled.csv",
            "--ranked-table", f"{outdir}/table_07_pareto_bootstrap.csv",
            "--ranked-criteria", f"{outdir}/fahp_ranked_criteria.txt",
            "--n-bootstrap", str(args.n_bootstrap),
            *common,
        ],
    )

    shared_cfg = [
        "--scaled-data", f"{outdir}/preprocessed_scaled.csv",
        "--selected-config", f"{outdir}/selected_configuration.json",
        "--fahp-weights", f"{outdir}/table_03_fahp_weights.csv",
        *common,
    ]
    _run("run_baseline_comparison.py", shared_cfg)
    _run("run_statistical_tests.py", ["--n-bootstrap", str(args.n_bootstrap), *shared_cfg])
    _run("run_ablation.py", shared_cfg)
    _run("run_fuzzy_validity.py", [
        "--scaled-data", f"{outdir}/preprocessed_scaled.csv",
        "--selected-config", f"{outdir}/selected_configuration.json",
        *common,
    ])
    topsis_extra = ["--topsis-data", args.imputed] if args.imputed else []
    _run("run_topsis_priorities.py", shared_cfg + topsis_extra)
    _run("run_topsis_sensitivity.py", shared_cfg + topsis_extra)
    _run("run_runtime_benchmark.py", shared_cfg)

    print(f"\nSemua tabel dan figure manuskrip ditulis ke '{outdir}/'.")


if __name__ == "__main__":
    main()
