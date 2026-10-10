"""Mengubah Data_Upload_Batik.xlsx (nama usaha + 15 indikator) menjadi data/batik_asli.csv
(msme_id + C1..C16) yang dibaca pipeline.

Catatan: Excel tidak memuat C10 (Motif Variations) karena konstan di seluruh data (naskah Bagian 3.2),
sehingga kolom C10 ditambahkan dengan nilai konstan 1. Setelah Min-Max, kolom ini = 0 dan tidak
memengaruhi jarak Euclidean.

Pakai:  python scripts/convert_excel_to_csv.py --xlsx Data_Upload_Batik.xlsx
"""
import argparse
from pathlib import Path

import pandas as pd

COLS_15 = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12", "C13", "C14", "C15", "C16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default="data/batik_asli.csv")
    a = ap.parse_args()
    d = pd.read_excel(a.xlsx)
    if d.shape[1] != 16:
        raise SystemExit(f"Diharapkan 16 kolom (nama + 15 indikator), ditemukan {d.shape[1]}.")
    vals = d.iloc[:, 1:].copy()
    vals.columns = COLS_15
    vals["C10"] = 1.0
    out = pd.DataFrame({"msme_id": [f"MSME_{i + 1:04d}" for i in range(len(d))]})
    for c in [f"C{i}" for i in range(1, 17)]:
        out[c] = vals[c].values
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.DataFrame({"msme_id": out["msme_id"], "nama_usaha": d.iloc[:, 0].values}).to_csv(
        Path(a.out).with_name("pemetaan_id_nama_usaha.csv"), index=False)
    print(f"OK: {len(out)} baris -> {a.out} (nilai hilang: {int(out.isna().sum().sum())})")


if __name__ == "__main__":
    main()
