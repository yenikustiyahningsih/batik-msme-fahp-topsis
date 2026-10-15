
"""Tahap 12: rata-rata waktu komputasi atas fit berulang -> Tabel 15.

Runtime bersifat spesifik-lingkungan dan digunakan untuk perbandingan
relatif, bukan sebagai klaim performa absolut.

Jika bobot FAHP Type-1 tidak tersedia, W-KM FAHP diberi status
not_evaluated. Bobot Type-2 tidak digunakan sebagai pengganti.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import (
    clara,
    fcm,
    it2fcm,
    load_baseline_params,
    weighted_kmeans,
)
from batik_fahp_topsis.clustering import fit_kmeans


def _time_it(fn, n_repeats: int = 30) -> tuple[float, float]:
    """Hitung rata-rata dan standar deviasi waktu eksekusi."""
    if n_repeats < 2:
        raise ValueError("--n-repeats harus minimal 2 untuk menghitung SD.")

    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)

    return (
        float(np.mean(durations)),
        float(np.std(durations, ddof=1)),
    )


def _valid_weights(values: np.ndarray) -> bool:
    """Periksa apakah seluruh bobot tersedia dan dapat dinormalisasi."""
    values = np.asarray(values, dtype=float)

    return (
        values.ndim == 1
        and values.size > 0
        and np.all(np.isfinite(values))
        and np.all(values >= 0)
        and values.sum() > 0
    )


def _add_timing(rows: list[dict], method: str, fn, n_repeats: int) -> None:
    """Ukur runtime metode dan tambahkan baris hasil."""
    mean_s, std_s = _time_it(fn, n_repeats)
    rows.append({
        "method": method,
        "mean_s": mean_s,
        "SD": std_s,
        "status": "evaluated",
        "note": "",
    })


def _add_unavailable(rows: list[dict], method: str, reason: str) -> None:
    """Catat metode yang tidak dapat diukur secara valid."""
    rows.append({
        "method": method,
        "mean_s": np.nan,
        "SD": np.nan,
        "status": "not_evaluated",
        "note": reason,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaled-data",
        default="outputs/preprocessed_scaled.csv",
    )
    parser.add_argument(
        "--selected-config",
        default="outputs/selected_configuration.json",
    )
    parser.add_argument(
        "--fahp-weights",
        default="outputs/table_03_fahp_weights.csv",
    )
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-repeats", type=int, default=30)
    args = parser.parse_args()

    if args.n_repeats < 2:
        parser.error("--n-repeats harus minimal 2.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(
        Path(args.selected_config).read_text(encoding="utf-8")
    )
    features = config["features"]
    k = int(config["k"])

    if not features:
        raise ValueError("Fitur pada selected configuration kosong.")

    missing_features = [
        feature for feature in features
        if feature not in df_scaled.columns
    ]
    if missing_features:
        raise ValueError(
            f"Fitur tidak ditemukan dalam scaled data: {missing_features}"
        )

    X = df_scaled[features].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Data fitur mengandung NaN atau nilai non-finite. "
            "Periksa hasil preprocessing sebelum benchmark."
        )

    weights_table = load_data(args.fahp_weights).set_index("criterion")
    missing_criteria = [
        feature for feature in features
        if feature not in weights_table.index
    ]
    if missing_criteria:
        raise ValueError(
            f"Fitur tidak ditemukan pada tabel bobot: {missing_criteria}"
        )

    required_columns = ["type1_fahp_weight", "type2_fahp_weight"]
    missing_columns = [
        column for column in required_columns
        if column not in weights_table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Kolom bobot tidak ditemukan: {missing_columns}"
        )

    type1_raw = weights_table.loc[
        features, "type1_fahp_weight"
    ].to_numpy(dtype=float)

    type2_raw = weights_table.loc[
        features, "type2_fahp_weight"
    ].to_numpy(dtype=float)

    type1_available = _valid_weights(type1_raw)
    type2_available = _valid_weights(type2_raw)

    if type2_available:
        type2_w = type2_raw / type2_raw.sum()
    else:
        raise ValueError(
            "Bobot FAHP Type-2 tidak tersedia atau tidak valid; "
            "benchmark W-KM T2 tidak dapat dijalankan."
        )

    params = load_baseline_params()
    rows = []

    _add_timing(
        rows, "Proposed",
        lambda: fit_kmeans(X, k),
        args.n_repeats,
    )
    _add_timing(
        rows, "CLARA",
        lambda: clara(X, k, params),
        args.n_repeats,
    )
    _add_timing(
        rows, "FCM",
        lambda: fcm(X, k, params),
        args.n_repeats,
    )
    _add_timing(
        rows, "IT2FCM",
        lambda: it2fcm(X, k, params),
        args.n_repeats,
    )

    if type1_available:
        type1_w = type1_raw / type1_raw.sum()
        _add_timing(
            rows, "W-KM FAHP",
            lambda: weighted_kmeans(X, type1_w, k, params),
            args.n_repeats,
        )
    else:
        _add_unavailable(
            rows,
            "W-KM FAHP",
            "Bobot FAHP Type-1 tidak tersedia atau mengandung NaN; "
            "tidak dilakukan substitusi dengan bobot Type-2.",
        )

    _add_timing(
        rows, "W-KM T2",
        lambda: weighted_kmeans(X, type2_w, k, params),
        args.n_repeats,
    )

    table15 = pd.DataFrame(rows)
    output_path = outdir / "table_15_runtime.csv"
    table15.to_csv(output_path, index=False)

    print(table15.to_string(index=False))
    print(f"\nRuntime benchmark tersimpan: {output_path}")

    if not type1_available:
        print(
            "\nPERINGATAN: W-KM FAHP tidak diukur karena bobot "
            "Type-1 tidak tersedia."
        )


if __name__ == "__main__":
    main()
