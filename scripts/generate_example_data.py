"""Membuat dataset contoh sintetis yang sudah dianonimkan dengan skema yang sama persis dengan
dataset MSME Batik asli yang bersifat rahasia (data/schema.md), sehingga seluruh pipeline dapat
dijalankan dari awal sampai akhir tanpa perlu akses ke data asli. Nilai-nilainya diambil dari
rentang yang masuk akal saja; tidak merepresentasikan MSME nyata mana pun dan tidak boleh
ditafsirkan sebagai data asli.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_rows: int = 848, n_missing: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = {
        "msme_id": [f"MSME_{i:04d}" for i in range(n_rows)],
        "C1": rng.integers(5, 500, n_rows).astype(float),        # Number of Consumers
        "C2": rng.integers(0, 20, n_rows).astype(float),         # Consumer Complaints
        "C3": rng.integers(0, 10, n_rows).astype(float),         # Partnerships
        "C4": rng.integers(0, 5, n_rows).astype(float),          # Customer Rewards
        "C5": rng.integers(0, 10, n_rows).astype(float),         # Employee Training
        "C6": rng.integers(0, 10, n_rows).astype(float),         # Product Branding
        "C7": rng.uniform(0, 1, n_rows) * 100,                   # Raw Material Price Increase (%, cost)
        "C8": rng.integers(0, 10, n_rows).astype(float),         # Owner Training
        "C9": rng.integers(0, 2, n_rows).astype(float),          # Business License (0/1)
        "C10": np.ones(n_rows),                                  # Motif Variations (konstan, lihat schema.md)
        "C11": rng.integers(0, 2, n_rows).astype(float),         # Offline Purchasing Rules
        "C12": rng.integers(0, 15, n_rows).astype(float),        # IT-Certified Employees
        "C13": rng.integers(6, 20, n_rows).astype(float),        # Owner Education (tahun)
        "C14": rng.integers(0, 5, n_rows).astype(float),         # Marketplace channels
        "C15": rng.integers(0, 2, n_rows).astype(float),         # Online Payment Facilities
        "C16": rng.integers(0, 2, n_rows).astype(float),         # Own MIS
    }
    df = pd.DataFrame(data)

    # Menyisipkan nilai hilang (MCAR) di seluruh 16 indikator agar sesuai dengan laporan
    # manuskrip "200 nilai yang hilang" (Bagian 3.2), sehingga imputasi KNN di
    # run_preprocessing.py punya sesuatu yang representatif untuk dikerjakan.
    criteria_cols = [c for c in df.columns if c.startswith("C")]
    flat_positions = rng.choice(len(df) * len(criteria_cols), size=n_missing, replace=False)
    for pos in flat_positions:
        row_idx = pos // len(criteria_cols)
        col_idx = pos % len(criteria_cols)
        df.loc[row_idx, criteria_cols[col_idx]] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/example_data.csv")
    parser.add_argument("--n-rows", type=int, default=848)
    parser.add_argument("--n-missing", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(n_rows=args.n_rows, n_missing=args.n_missing, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Menulis {len(df)} baris ke {out_path}")


if __name__ == "__main__":
    main()
