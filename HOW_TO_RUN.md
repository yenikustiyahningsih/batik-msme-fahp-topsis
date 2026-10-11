# How to run (Python scripts from the terminal; not Streamlit or a notebook)

There are two ways. **Option A is for everyone** (uses the anonymized example data). **Option B is
for the authors only** (uses the confidential original data, which is not released).

## 1. Install (once)

Use **Python 3.11** (the version used for the manuscript results: 3.11.9). Python 3.12 or newer is
NOT recommended because `scikit-fuzzy 0.4.2` needs the `imp` and `distutils` modules, which have
been removed.

    python -m venv .venv
    .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
    pip install -r requirements-pinned.txt

- `requirements-pinned.txt`: the exact versions used in the manuscript. **Use this to reproduce the results.**
- `requirements.txt`: minimum version bounds, for general use only; numbers may shift slightly.

On Windows, put the venv in a short path (for example `python -m venv C:\vb`) if you get a
"filename too long" error.

## 2. Option A: example data (public)

    python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs

The example data are anonymized synthetic data (848 rows, 16 indicators) so that the code can run
from start to finish. **The numbers are NOT the same as in the manuscript**; only the workflow is
the same. With the example data, tables that contain business names are not produced.

## 3. Option B: original data (authors only)

    python scripts/convert_excel_to_csv.py --xlsx Data_Upload_Batik.xlsx
    python scripts/generate_all_tables.py --data data/batik_asli.csv --imputed data/batik_imputed_frozen.csv --outdir outputs --weights-csv params/paper_table3_weights.csv

The run takes about 1 to 5 minutes and prints nothing during the bootstrap stage; do not press
Ctrl+C. All tables (`table_03` to `table_15`; file number = manuscript table number) and figures
(`figure_*.png`) appear in the `outputs/` folder.

Key numbers that should appear for Option B: K=5 with 7 features; ARI 0.9762 (SD 0.0340) and NMI
0.9709; SBS (A9) SC 0.5747 / DBI 0.9047 / CHI 510.0; Friedman statistic for DBI 84.08 (Windows).
Computation time (Table 15) depends on the machine.

## Important notes
- `--weights-csv` uses the Type-2 FAHP weights from Table 3 of the manuscript. Without this option
  the program uses RANDOM expert matrices (a placeholder), so the weights and selected features
  will not match the manuscript.
- The experts' individual pairwise comparison matrices are not released. The conventional (Type-1)
  FAHP weights are therefore set equal to the Type-2 weights in this code, so the "conventional
  FAHP vs Type-2" comparison (Tables 8, 9, 10) does not provide evidence of a Type-2 advantage.
  This is also stated in the manuscript (Section 4.10).
- The original data have no C10 column (constant); the column is added automatically as a constant
  and excluded from the random subsets and SBS (15 nonconstant indicators).
- `--imputed data/batik_imputed_frozen.csv` uses the frozen KNN imputation result so that the
  numbers are the same on every computer. Without it, many neighbors are equidistant, so the
  imputation can pick different neighbors depending on row order or platform (SC for K=5 shifts
  between 0.4593 and 0.4609).
- TOPSIS (Tables 13 and 14) uses the decision matrix on the original 1 to 4 scale; clustering uses
  Min-Max data. Table 14 is computed per cluster (weight perturbation with random signs, 100
  repetitions, seed 42).
- All baseline hyperparameters are in `params/baseline_params.yaml`.