
"""Tahap 10: prioritas TOPSIS khusus per-cluster -> Tabel 12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import load_data
from batik_fahp_topsis.clustering import fit_kmeans
from batik_fahp_topsis.topsis import cluster_specific_topsis

COST_CRITERIA = ["C7"]


def count_tied_rows(values) -> int:
    """Menghitung jumlah baris yang nilainya muncul lebih dari sekali."""
    frequencies = values.value_counts()
    return int((values.map(frequencies) > 1).sum())


def main():
    parser = argparse.ArgumentParser(
        description="Menghitung prioritas TOPSIS dalam setiap cluster."
    )
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
    parser.add_argument(
        "--topsis-data",
        default=None,
        help=(
            "CSV indikator untuk matriks keputusan TOPSIS. "
            "Gunakan skala yang sesuai dengan metodologi manuskrip. "
            "Tanpa opsi ini, data Min-Max digunakan."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k harus minimal 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Muat data clustering dan konfigurasi.
    df_scaled = load_data(args.scaled_data)

    if "msme_id" not in df_scaled.columns:
        raise ValueError("Kolom 'msme_id' tidak ditemukan pada data clustering.")
    if df_scaled["msme_id"].isna().any():
        raise ValueError("Ditemukan msme_id kosong pada data clustering.")
    if df_scaled["msme_id"].duplicated().any():
        raise ValueError("Ditemukan msme_id duplikat pada data clustering.")

    config = json.loads(
        Path(args.selected_config).read_text(encoding="utf-8")
    )
    features = config["features"]
    k = int(config["k"])

    if not features:
        raise ValueError("Konfigurasi tidak memiliki fitur.")
    if len(features) != len(set(features)):
        raise ValueError("Daftar fitur mengandung duplikat.")
    if k < 2:
        raise ValueError("Jumlah cluster K harus minimal 2.")

    missing_features = set(features) - set(df_scaled.columns)
    if missing_features:
        raise ValueError(
            f"Fitur clustering tidak ditemukan: {sorted(missing_features)}"
        )

    X = df_scaled[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(
            "Data clustering mengandung NaN atau nilai tak hingga."
        )

    # 2. Muat dan validasi bobot Type-2 FAHP.
    weights_table = load_data(args.fahp_weights)

    required_weight_columns = {"criterion", "type2_fahp_weight"}
    missing_weight_columns = required_weight_columns - set(
        weights_table.columns
    )
    if missing_weight_columns:
        raise ValueError(
            "Kolom tabel bobot tidak ditemukan: "
            f"{sorted(missing_weight_columns)}"
        )

    if weights_table["criterion"].duplicated().any():
        raise ValueError("Terdapat kriteria duplikat pada tabel bobot.")

    weights_table = weights_table.set_index("criterion")

    missing_weights = set(features) - set(weights_table.index)
    if missing_weights:
        raise ValueError(
            f"Bobot tidak tersedia untuk fitur: {sorted(missing_weights)}"
        )

    w = weights_table.loc[
        features, "type2_fahp_weight"
    ].to_numpy(dtype=float)

    if not np.isfinite(w).all() or (w <= 0).any() or w.sum() <= 0:
        raise ValueError("Bobot Type-2 FAHP harus finite dan positif.")

    w = w / w.sum()

    # 3. Clustering tetap dihitung menggunakan data Min-Max.
    _, labels = fit_kmeans(X, k)

    # 4. Tentukan data matriks keputusan TOPSIS.
    df_topsis = df_scaled

    if args.topsis_data:
        df_topsis = load_data(args.topsis_data)

        if "msme_id" not in df_topsis.columns:
            raise ValueError("Kolom 'msme_id' tidak ditemukan pada data TOPSIS.")
        if df_topsis["msme_id"].isna().any():
            raise ValueError("Ditemukan msme_id kosong pada data TOPSIS.")
        if df_topsis["msme_id"].duplicated().any():
            raise ValueError("Ditemukan msme_id duplikat pada data TOPSIS.")

        if list(df_topsis["msme_id"]) != list(df_scaled["msme_id"]):
            raise ValueError(
                "ID pada --topsis-data tidak sama atau urutannya berbeda "
                "dari --scaled-data."
            )

        missing_topsis = set(features) - set(df_topsis.columns)
        if missing_topsis:
            raise ValueError(
                f"Fitur TOPSIS tidak ditemukan: {sorted(missing_topsis)}"
            )

        X_topsis = df_topsis[features].to_numpy(dtype=float)
        if not np.isfinite(X_topsis).all():
            raise ValueError(
                "Data TOPSIS mengandung NaN atau nilai tak hingga."
            )

    # 5. Hitung TOPSIS secara terpisah dalam setiap cluster.
    priorities = cluster_specific_topsis(
        df_topsis,
        labels,
        features,
        w,
        COST_CRITERIA,
    )

    # 6. Tambahkan nama usaha jika tersedia.
    names_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pemetaan_id_nama_usaha.csv"
    )

    if names_path.exists():
        names = load_data(names_path)

        if "msme_id" not in names.columns:
            raise ValueError(
                "Kolom 'msme_id' tidak ditemukan pada file pemetaan."
            )
        if names["msme_id"].isna().any():
            raise ValueError("File pemetaan memiliki msme_id kosong.")
        if names["msme_id"].duplicated().any():
            raise ValueError(
                "File pemetaan nama usaha memiliki msme_id duplikat."
            )

        priorities = priorities.merge(
            names,
            on="msme_id",
            how="left",
            validate="many_to_one",
        )

    # 7. Validasi hasil TOPSIS.
    required_columns = {
        "msme_id",
        "cluster",
        "topsis_score",
        "rank_within_cluster",
    }
    missing_columns = required_columns - set(priorities.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom keluaran TOPSIS tidak ditemukan: {sorted(missing_columns)}"
        )
    if priorities["msme_id"].duplicated().any():
        raise ValueError("Output TOPSIS memiliki msme_id duplikat.")
    if priorities["topsis_score"].isna().any():
        raise ValueError("Ditemukan skor TOPSIS kosong.")
    if priorities["rank_within_cluster"].isna().any():
        raise ValueError("Ditemukan peringkat TOPSIS kosong.")
    if not np.isfinite(
        priorities["topsis_score"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Skor TOPSIS mengandung nilai tak hingga.")

    # 8. Diagnosis ties dalam setiap cluster.
    rank_tie_mask = priorities.groupby(
        "cluster", sort=False
    )["rank_within_cluster"].transform(count_tied_rows) > 1

    exact_score_tie_mask = priorities.groupby(
        "cluster", sort=False
    )["topsis_score"].transform(count_tied_rows) > 1

    rounded_scores = priorities["topsis_score"].round(10)
    rounded_score_tie_mask = rounded_scores.groupby(
        priorities["cluster"], sort=False
    ).transform(count_tied_rows) > 1

    print(f"Konfigurasi clustering: K={k}, fitur={len(features)}")
    print(
        "UMKM yang berbagi peringkat dalam cluster: "
        f"{int(rank_tie_mask.sum())}"
    )
    print(
        "UMKM dengan skor TOPSIS identik secara numerik: "
        f"{int(exact_score_tie_mask.sum())}"
    )
    print(
        "UMKM dengan skor sama setelah pembulatan 10 desimal: "
        f"{int(rounded_score_tie_mask.sum())}"
    )

    # 9. Simpan hasil lengkap dan top-K.
    # Ranking seri pada batas K dapat menghasilkan lebih dari K baris.
    top_k = priorities[
        priorities["rank_within_cluster"] <= args.top_k
    ].copy()

    top_k.to_csv(
        outdir / "table_13_topsis_priorities.csv",
        index=False,
    )
    priorities.to_csv(
        outdir / "topsis_priorities_full.csv",
        index=False,
    )

    print(f"Jumlah baris tabel top-K: {len(top_k)}")
    print(f"Hasil lengkap: {outdir / 'topsis_priorities_full.csv'}")
    print(top_k.to_string(index=False))


if __name__ == "__main__":
    main()
