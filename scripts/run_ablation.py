
"""Tahap 8: analisis ablasi -> Tabel 10.

A4 dan A6 membutuhkan bobot FAHP Type-1. Jika bobot tersebut tidak
tersedia atau mengandung NaN, kedua konfigurasi ditandai not_evaluated.
Bobot Type-2 tidak digunakan sebagai pengganti bobot Type-1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_data
from batik_fahp_topsis.baselines import load_baseline_params, weighted_kmeans
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.feature_selection import (
    random_subsets,
    sequential_backward_selection,
)
from batik_fahp_topsis.preprocessing import ALL_CRITERIA
from batik_fahp_topsis.validity_metrics import internal_validity


def normalize_weights(values: np.ndarray, name: str) -> np.ndarray:
    """Validasi dan normalisasi bobot yang tersedia."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} harus berupa array satu dimensi.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} mengandung NaN atau nilai non-finite.")

    if np.any(values < 0):
        raise ValueError(f"{name} mengandung bobot negatif.")

    total = values.sum()
    if total <= 0:
        raise ValueError(f"Total {name} harus lebih besar dari nol.")

    return values / total


def type1_weights_available(values: np.ndarray) -> bool:
    """True jika seluruh bobot Type-1 tersedia dan dapat digunakan."""
    values = np.asarray(values, dtype=float)
    return (
        values.ndim == 1
        and len(values) > 0
        and np.all(np.isfinite(values))
        and np.all(values >= 0)
        and values.sum() > 0
    )


def add_evaluated_row(
    rows: list[dict],
    variant: str,
    setting: str,
    X: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Tambahkan hasil konfigurasi yang berhasil dievaluasi."""
    rows.append({
        "variant": variant,
        "setting": setting,
        **internal_validity(X, labels),
        "status": "evaluated",
        "note": "",
    })


def add_unavailable_row(
    rows: list[dict],
    variant: str,
    setting: str,
    reason: str,
) -> None:
    """Catat konfigurasi yang tidak dapat dievaluasi tanpa mengarang hasil."""
    rows.append({
        "variant": variant,
        "setting": setting,
        "SC": np.nan,
        "DBI": np.nan,
        "CHI": np.nan,
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
    parser.add_argument("--n-random-subsets", type=int, default=100)
    args = parser.parse_args()

    if args.n_random_subsets < 1:
        parser.error("--n-random-subsets harus minimal 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_scaled = load_data(args.scaled_data)
    config = json.loads(
        Path(args.selected_config).read_text(encoding="utf-8")
    )

    features = config["features"]
    k = int(config["k"])
    n_features = len(features)

    if not features:
        raise ValueError("Daftar fitur pada selected configuration kosong.")

    if len(set(features)) != len(features):
        raise ValueError("Daftar fitur mengandung duplikasi.")

    missing_features = [
        feature for feature in ALL_CRITERIA
        if feature not in df_scaled.columns
    ]
    if missing_features:
        raise ValueError(
            f"Fitur ALL_CRITERIA tidak ditemukan dalam scaled data: "
            f"{missing_features}"
        )

    missing_selected = [
        feature for feature in features
        if feature not in df_scaled.columns
    ]
    if missing_selected:
        raise ValueError(
            f"Fitur terpilih tidak ditemukan dalam scaled data: "
            f"{missing_selected}"
        )

    weights_table = load_data(args.fahp_weights).set_index("criterion")

    missing_weight_criteria = [
        feature for feature in ALL_CRITERIA
        if feature not in weights_table.index
    ]
    if missing_weight_criteria:
        raise ValueError(
            f"Kriteria tidak ditemukan dalam tabel bobot FAHP: "
            f"{missing_weight_criteria}"
        )

    required_weight_columns = [
        "type1_fahp_weight",
        "type2_fahp_weight",
    ]
    missing_weight_columns = [
        column for column in required_weight_columns
        if column not in weights_table.columns
    ]
    if missing_weight_columns:
        raise ValueError(
            f"Kolom bobot tidak ditemukan: {missing_weight_columns}"
        )

    # Ambil bobot Type-1 tanpa menormalisasi NaN.
    # Bobot yang tidak tersedia tidak boleh diganti dengan Type-2.
    selected_type1_raw = weights_table.loc[
        features, "type1_fahp_weight"
    ].to_numpy(dtype=float)

    full_type1_raw = weights_table.loc[
        list(ALL_CRITERIA), "type1_fahp_weight"
    ].to_numpy(dtype=float)

    selected_type1_available = type1_weights_available(
        selected_type1_raw
    )
    full_type1_available = type1_weights_available(full_type1_raw)

    # Type-2 harus tersedia karena A5 dan A7 memerlukannya.
    selected_type2_w = normalize_weights(
        weights_table.loc[
            features, "type2_fahp_weight"
        ].to_numpy(dtype=float),
        "bobot Type-2 pada fitur terpilih",
    )
    full_type2_w = normalize_weights(
        weights_table.loc[
            list(ALL_CRITERIA), "type2_fahp_weight"
        ].to_numpy(dtype=float),
        "bobot Type-2 pada seluruh fitur",
    )

    params = load_baseline_params()
    rows = []

    # A1: seluruh 16 indikator.
    X_all = df_scaled[list(ALL_CRITERIA)].to_numpy(dtype=float)
    _, labels_all = fit_kmeans(X_all, k)
    add_evaluated_row(
        rows, "A1", "All 16", X_all, labels_all
    )

    # A2 / A3: subset yang sama sesuai implementasi pipeline saat ini.
    X_sel = df_scaled[features].to_numpy(dtype=float)
    _, labels_selected = fit_kmeans(X_sel, k)

    add_evaluated_row(
        rows, "A2", "FAHP FS7", X_sel, labels_selected
    )
    add_evaluated_row(
        rows, "A3", "Type-2 FS7", X_sel, labels_selected
    )

    # A4: bobot FAHP Type-1 pada seluruh 16 fitur.
    if full_type1_available:
        full_type1_w = normalize_weights(
            full_type1_raw, "bobot Type-1 seluruh fitur"
        )
        labels_a4, _ = weighted_kmeans(
            X_all, full_type1_w, k, params
        )
        add_evaluated_row(
            rows, "A4", "FAHP weighted 16", X_all, labels_a4
        )
    else:
        add_unavailable_row(
            rows,
            "A4",
            "FAHP weighted 16",
            "Bobot FAHP Type-1 tidak tersedia atau mengandung "
            "NaN; konfigurasi tidak dievaluasi.",
        )

    # A5: bobot FAHP Type-2 pada seluruh 16 fitur.
    labels_a5, _ = weighted_kmeans(
        X_all, full_type2_w, k, params
    )
    add_evaluated_row(
        rows, "A5", "Type-2 weighted 16", X_all, labels_a5
    )

    # A6: subset terpilih + bobot Type-1.
    if selected_type1_available:
        selected_type1_w = normalize_weights(
            selected_type1_raw, "bobot Type-1 fitur terpilih"
        )
        labels_a6, _ = weighted_kmeans(
            X_sel, selected_type1_w, k, params
        )
        add_evaluated_row(
            rows, "A6", "FS7 + FAHP weight", X_sel, labels_a6
        )
    else:
        add_unavailable_row(
            rows,
            "A6",
            "FS7 + FAHP weight",
            "Bobot FAHP Type-1 untuk fitur terpilih tidak tersedia "
            "atau mengandung NaN; konfigurasi tidak dievaluasi.",
        )

    # A7: subset terpilih + bobot Type-2.
    labels_a7, _ = weighted_kmeans(
        X_sel, selected_type2_w, k, params
    )
    add_evaluated_row(
        rows, "A7", "FS7 + Type-2 weight", X_sel, labels_a7
    )

    # A8: rata-rata dari subset acak berisi jumlah fitur yang sama.
    nonconstant = [
        column for column in ALL_CRITERIA
        if column != "C10"
    ]
    if n_features > len(nonconstant):
        raise ValueError(
            f"Jumlah fitur terpilih ({n_features}) melebihi "
            f"jumlah fitur nonkonstan ({len(nonconstant)})."
        )

    subsets = random_subsets(
        nonconstant,
        subset_size=n_features,
        n_subsets=args.n_random_subsets,
    )

    sc_list, dbi_list, chi_list = [], [], []
    for subset in subsets:
        X_random = df_scaled[subset].to_numpy(dtype=float)
        _, labels_random = fit_kmeans(X_random, k)
        metrics = internal_validity(X_random, labels_random)

        sc_list.append(metrics["SC"])
        dbi_list.append(metrics["DBI"])
        chi_list.append(metrics["CHI"])

    rows.append({
        "variant": "A8",
        "setting": "Random FS7 mean",
        "SC": float(np.mean(sc_list)),
        "DBI": float(np.mean(dbi_list)),
        "CHI": float(np.mean(chi_list)),
        "status": "evaluated",
        "note": f"Rata-rata {len(subsets)} subset acak.",
    })

    # A9: sequential backward selection berbasis data.
    X_nonconst = df_scaled[nonconstant].to_numpy(dtype=float)
    sbs_features = sequential_backward_selection(
        X_nonconst,
        nonconstant,
        k,
        floor_n_features=n_features,
    )
    X_sbs = df_scaled[sbs_features].to_numpy(dtype=float)
    _, labels_sbs = fit_kmeans(X_sbs, k)

    add_evaluated_row(
        rows, "A9", "SBS FS7", X_sbs, labels_sbs
    )

    table9 = pd.DataFrame(rows)
    output_path = outdir / "table_10_ablation.csv"
    table9.to_csv(output_path, index=False)

    print(table9.to_string(index=False))
    print(f"\nTabel ablasi tersimpan: {output_path}")

    unavailable = table9.loc[
        table9["status"] == "not_evaluated", "variant"
    ].tolist()
    if unavailable:
        print(
            "\nPERINGATAN: konfigurasi berikut tidak dievaluasi "
            f"karena bobot Type-1 tidak tersedia: {unavailable}"
        )


if __name__ == "__main__":
    main()
