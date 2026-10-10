# Joint Feature and Cluster Optimization with Type-2 FAHP-Guided Feature Selection and TOPSIS for Robust Batik MSME Prioritization

[Bahasa Indonesia](README.id.md) | English

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23287436.svg)](https://doi.org/10.5281/zenodo.23287436)

Reproducibility repository for manuscript **Paper ID 20265381** (IJIES / INASS).

This repository contains the computational pipeline used to produce the tables and figures of the
manuscript: preprocessing, Type-2 FAHP criterion weighting, joint feature-cluster optimization
(90-configuration grid), Pareto efficiency and mean-rank selection, bootstrap stability analysis,
baseline comparison (CLARA, FCM, IT2FCM, weighted K-Means), statistical significance tests,
ablation analysis, fuzzy-specific validity, cluster-specific TOPSIS, and TOPSIS sensitivity analysis.

> **Data confidentiality.** The original Batik MSME data (848 MSMEs, 16 indicators) are commercially
> confidential and are **not** included (see the Data Availability statement in the manuscript).
> The repository provides an **anonymized synthetic example dataset** (`data/example_data.csv`,
> created by `scripts/generate_example_data.py`) with the same schema and plausible value ranges,
> so the whole pipeline can be run end to end without the original data. Results on the example
> data **do not match the manuscript numbers**; only the workflow is identical. To reproduce the
> manuscript numbers, an authorized copy of the original data with the same schema
> (see `data/schema.md`) is required.

## Quick start

Use **Python 3.11** (the manuscript results were produced with 3.11.9). Python 3.12 or newer is not
supported because `scikit-fuzzy 0.4.2` depends on the removed `imp` and `distutils` modules.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-pinned.txt

# Optional: regenerate the anonymized example dataset
python scripts/generate_example_data.py

# Run the whole pipeline on the example data
python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs
```

- `requirements-pinned.txt`: exact versions used in the manuscript. Use this to reproduce results.
- `requirements.txt`: minimum version bounds for general use; numbers may shift slightly.

The run takes a few minutes and prints little during the bootstrap stage; do not interrupt it.
Each `outputs/table_*.csv` corresponds to one manuscript table and each `outputs/figure_*.png` to
one figure. `outputs/results_90_configs.csv` is the single authoritative results file from which
all other tables and figures are derived. A detailed run guide (in Indonesian) is in
`CARA_MENJALANKAN.md`.

## Repository structure

```
batik-msme-fahp-topsis/
├── README.md / README.id.md
├── CARA_MENJALANKAN.md            # detailed run guide (Indonesian)
├── requirements.txt               # minimum version bounds
├── requirements-pinned.txt        # exact versions used in the manuscript
├── data/
│   ├── schema.md                  # data dictionary for C1-C16, imputation and scaling rules
│   └── example_data.csv           # anonymized synthetic example (not real MSME data)
├── params/
│   ├── baseline_params.yaml       # baseline hyperparameters, seeds, library versions
│   ├── fahp_pairwise_template.yaml# template for the four experts' pairwise matrices
│   └── paper_table3_weights.csv   # Type-2 FAHP weights reported in Table 3
├── src/batik_fahp_topsis/
│   ├── preprocessing.py           # KNN imputation, Min-Max scaling, data-quality audit
│   ├── fahp.py                    # Type-2 FAHP and conventional (Type-1) FAHP weighting
│   ├── feature_selection.py       # FAHP-ranked subsets, random subsets, SBS
│   ├── validity_metrics.py        # SC, DBI, CHI, PC, PE, XB, FOU
│   ├── clustering.py              # joint K-Means grid search
│   ├── baselines.py               # CLARA, FCM, IT2FCM, weighted K-Means (see note below)
│   ├── pareto.py                  # Pareto efficiency + equal-weight mean-rank selection
│   ├── bootstrap.py               # bootstrap resampling, ARI/NMI stability
│   ├── statistical_tests.py       # Friedman, Wilcoxon with Holm correction, rank-biserial effect size
│   ├── topsis.py                  # cluster-specific TOPSIS + sensitivity / stress test
│   └── reproducibility.py         # centralized seed control
├── scripts/                       # one run_*.py per analysis stage, plus generate_all_tables.py
├── tests/test_metrics.py
└── outputs/                       # created at runtime
```

## Mapping to the manuscript

| Manuscript item | Script | Output file(s) |
|---|---|---|
| Tables 3, 4 (Type-2 FAHP weights, expert CR) | `run_fahp_weights.py` | `table_03_fahp_weights.csv`, `table_04_expert_cr.csv` |
| Tables 6, 7 (Pareto set, mean rank, bootstrap ARI/NMI + SD) | `run_pareto_and_ranking.py`, `run_bootstrap_stability.py` | `table_06_leading_configs.csv`, `table_07_pareto_bootstrap.csv`, `table_07_bootstrap_per_repetition.csv` |
| Figure 2 (composite-score heatmap) | `run_joint_optimization.py` | `figure_02_composite_score.png` |
| Figure 3 (trade-off scatter) | `run_pareto_and_ranking.py` | `figure_03_tradeoff.png` |
| Table 8 (baseline comparison) | `run_baseline_comparison.py` | `table_08_baseline_comparison.csv` |
| Table 9 (Friedman / Wilcoxon-Holm) | `run_statistical_tests.py` | `table_09_statistical_tests.csv`, `table_09_holm_wilcoxon_{SC,DBI,CHI}.csv` |
| Table 10 (ablation) | `run_ablation.py` | `table_10_ablation.csv` |
| Table 11 (fuzzy-specific validity) | `run_fuzzy_validity.py` | `table_11_fuzzy_validity.csv` |
| Table 12 (final cluster distribution) | `run_bootstrap_stability.py` | `table_12_cluster_distribution.csv` |
| Table 13 (TOPSIS priorities) | `run_topsis_priorities.py` | `table_13_topsis_priorities.csv` |
| Table 14, Figure 6 (TOPSIS sensitivity) | `run_topsis_sensitivity.py` | `table_14_topsis_sensitivity.csv`, `figure_06_topsis_sensitivity.png` |
| Table 15 (runtime) | `run_runtime_benchmark.py` | `table_15_runtime.csv` |

On the example data, tables that contain business names (for example Table 13) are not produced.

## Baseline parameters

All baseline hyperparameters, initialization schemes, random-seed rules, and library versions are
recorded in `params/baseline_params.yaml` and read directly by
`src/batik_fahp_topsis/baselines.py`, so no baseline setting is hard-coded in the analysis scripts.

## Random seeds and reproducibility

`src/batik_fahp_topsis/reproducibility.py` centralizes every random seed (`random_state = 42` for
the deterministic full-data grid; distinct, recorded seeds for each bootstrap repetition).
`scripts/generate_all_tables.py` is the single authoritative entry point: it calls each stage in
order from the same `results_90_configs.csv`, so Pareto status, ranks, the selected configuration,
bootstrap statistics, and every manuscript table are generated automatically from one results
file rather than transcribed by hand between stages.

## Notes

- Table numbering: `table_NN_*.csv` follows the manuscript table numbers. Tables 1, 2, and 5
  (related work, missing values, experimental settings) are not program outputs; experimental
  settings are in `params/baseline_params.yaml`. Figures 1, 4, and 5 are not generated by this
  repository.
- SBS (Table 10, A9) uses the 15 nonconstant indicators; C10 (zero variance) is excluded, as for
  the random subsets (A8).
- IT2FCM uses a simplified type reduction (averaging the lower and upper memberships) rather than
  full Karnik-Mendel; CLARA uses a self-written PAM-style medoid solver. Both are described in
  `params/baseline_params.yaml`.
- `run_dbscan()` remains in `baselines.py` only as a utility. DBSCAN was removed from the manuscript
  and is not called by the pipeline.
- The original expert pairwise matrices are not released. The conventional (Type-1) FAHP weights
  are therefore set equal to the Type-2 weights in this code, so the "conventional FAHP versus
  Type-2" comparison (Tables 8, 9, 10) does not by itself show an advantage of Type-2. This is also
  stated in the manuscript (Section 4.10).
- Friedman statistics and runtimes can differ slightly across operating systems and machines. The
  manuscript numbers were produced on Windows with Python 3.11.9 and the versions in
  `requirements-pinned.txt`.

## Citation

If you use this code, please cite the Zenodo archive: https://doi.org/10.5281/zenodo.23287436

## License

MIT License (see `LICENSE`). The anonymized example dataset is synthetic and may be freely
distributed; the original MSME dataset is confidential and is not included here.