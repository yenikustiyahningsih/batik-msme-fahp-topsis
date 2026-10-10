"""Preprocessing: audit kualitas data, imputasi KNN, penskalaan Min-Max (manuskrip Bagian 3.2)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

ALL_CRITERIA: Sequence[str] = [f"C{i}" for i in range(1, 17)]


def audit_missing_values(df: pd.DataFrame, criteria: Sequence[str] = ALL_CRITERIA) -> dict:
    """Menghitung jumlah nilai yang hilang per kolom dan totalnya; menandai kolom yang konstan.

    Mengembalikan dict ringkasan yang dipakai untuk mengisi paragraf audit-kualitas-data di
    Bagian 3.2 (mis. "Dataset yang digunakan memiliki 200 nilai yang hilang").
    """
    missing_per_column = df[list(criteria)].isna().sum().to_dict()
    total_missing = int(sum(missing_per_column.values()))
    constant_columns = [c for c in criteria if df[c].nunique(dropna=True) <= 1]
    return {
        "missing_per_column": missing_per_column,
        "total_missing": total_missing,
        "constant_columns": constant_columns,
    }


def impute_knn(df: pd.DataFrame, criteria: Sequence[str] = ALL_CRITERIA, k: int = 5) -> pd.DataFrame:
    """Mengimputasi nilai indikator yang hilang dengan K-nearest neighbors (k = 5), pada skala asli.

    Sesuai manuskrip Bagian 3.2: "diimputasi menggunakan K-nearest neighbors dengan k sama dengan 5".
    """
    out = df.copy()
    imputer = KNNImputer(n_neighbors=k)
    out[list(criteria)] = imputer.fit_transform(out[list(criteria)])
    return out


def minmax_scale(df: pd.DataFrame, criteria: Sequence[str] = ALL_CRITERIA) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Menskalakan setiap indikator ke rentang [0, 1] dengan Min-Max. Mengembalikan frame yang
    sudah diskalakan beserta scaler yang sudah di-fit (scaler yang sama ini harus dipakai ulang
    saat inferensi/validasi, jangan pernah di-fit ulang pada resample, agar ruang evaluasi tetap
    identik untuk semua metode berbasis jarak)."""
    out = df.copy()
    scaler = MinMaxScaler()
    out[list(criteria)] = scaler.fit_transform(out[list(criteria)])
    return out, scaler


def preprocess(df: pd.DataFrame, criteria: Sequence[str] = ALL_CRITERIA, k: int = 5):
    """Pipeline preprocessing lengkap: audit -> imputasi KNN -> penskalaan Min-Max.

    Mengembalikan (scaled_df, audit_summary, scaler).
    """
    audit = audit_missing_values(df, criteria)
    imputed = impute_knn(df, criteria, k=k)
    scaled, scaler = minmax_scale(imputed, criteria)
    return scaled, audit, scaler
