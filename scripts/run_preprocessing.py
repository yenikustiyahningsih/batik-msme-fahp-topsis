"""Tahap 1: audit kualitas data, imputasi KNN, penskalaan Min-Max (tabel/paragraf, Bagian 3.2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_data
from batik_fahp_topsis.preprocessing import ALL_CRITERIA, audit_missing_values, minmax_scale, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/example_data.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--imputed",
        default=None,
        help="CSV hasil imputasi KNN yang sudah DIBEKUKAN (skala asli), mis. data/batik_imputed_frozen.csv. "
        "Jika diberikan, imputasi KNN dilewati dan hanya Min-Max yang dijalankan. Ini membuat hasil identik "
        "di semua komputer, karena imputasi KNN sensitif terhadap urutan baris saat jarak seri.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    if args.imputed:
        audit = audit_missing_values(df, ALL_CRITERIA)  # audit tetap dari data mentah
        imputed = load_data(args.imputed)
        assert imputed[list(ALL_CRITERIA)].isna().sum().sum() == 0, "File imputed masih punya NaN"
        assert list(imputed["msme_id"]) == list(df["msme_id"]), "Urutan/ID baris tidak sama dengan --data"
        scaled, _ = minmax_scale(imputed, ALL_CRITERIA)
        print(f"Memakai imputasi beku dari {args.imputed} (imputasi KNN dilewati)")
    else:
        scaled, audit, _ = preprocess(df, ALL_CRITERIA, k=5)

    scaled.to_csv(outdir / "preprocessed_scaled.csv", index=False)
    with open(outdir / "data_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"Total nilai hilang yang diimputasi: {audit['total_missing']}")
    print(f"Kolom konstan: {audit['constant_columns']}")
    print(f"Menulis {outdir / 'preprocessed_scaled.csv'}")


if __name__ == "__main__":
    main()
