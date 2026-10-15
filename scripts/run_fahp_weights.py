
"""Tahap 2: pembobotan FAHP Type-2 dan FAHP konvensional -> Tabel 3 dan Tabel 4.

Mendukung dua mode:
1. --weights-csv: menggunakan bobot yang sudah tersedia.
2. Tanpa --weights-csv: menghitung bobot dari matriks contoh sintetis.

Catatan:
- Matriks contoh sintetis bukan pengganti matriks penilaian pakar penelitian.
- Bobot Type-1 yang tidak tersedia disimpan sebagai NaN, bukan disamakan
  dengan Type-2.
- CR yang disalin dari manuskrip ditandai sebagai nilai yang tidak dihitung ulang.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import EXPERT_SEEDS, example_expert_matrix
from batik_fahp_topsis.fahp import (
    aggregate_group_matrix,
    conventional_fahp_weights,
    consistency_ratio,
    rank_criteria,
    type2_fahp_weights,
)
from batik_fahp_topsis.preprocessing import ALL_CRITERIA


def _validate_weights(
    weights: np.ndarray,
    name: str,
    require_positive: bool = True,
) -> np.ndarray:
    """Validasi dan normalisasi vektor bobot."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} harus berupa vektor satu dimensi nonkosong.")

    if not np.isfinite(weights).all():
        raise ValueError(f"{name} mengandung NaN atau nilai tak hingga.")

    if require_positive and (weights <= 0).any():
        raise ValueError(f"{name} harus berisi bobot positif.")
    if not require_positive and (weights < 0).any():
        raise ValueError(f"{name} tidak boleh berisi bobot negatif.")

    total = weights.sum()
    if total <= 0:
        raise ValueError(f"Jumlah {name} harus lebih besar dari nol.")

    return weights / total


def _validate_criteria(criteria: list[str]) -> None:
    """Memastikan definisi kriteria valid."""
    if not criteria:
        raise ValueError("ALL_CRITERIA tidak boleh kosong.")

    if len(criteria) != len(set(criteria)):
        raise ValueError("ALL_CRITERIA mengandung kriteria duplikat.")


def _rank_or_nan(weights: np.ndarray) -> np.ndarray:
    """Menghasilkan ranking jika semua bobot tersedia; selain itu NaN."""
    weights = np.asarray(weights, dtype=float)

    if not np.isfinite(weights).all():
        return np.full(len(weights), np.nan)

    return (
        pd.Series(weights)
        .rank(ascending=False, method="min")
        .to_numpy(dtype=float)
    )


def _from_weights_csv(path: str | Path, outdir: Path) -> None:
    """Membuat tabel dari bobot yang sudah tersedia."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"File bobot tidak ditemukan: {source_path}")

    weights_df = pd.read_csv(source_path)

    required_columns = {"criterion", "type2_fahp_weight"}
    missing_columns = required_columns - set(weights_df.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada CSV bobot: "
            f"{sorted(missing_columns)}"
        )

    if weights_df["criterion"].isna().any():
        raise ValueError("Kolom criterion mengandung nilai kosong.")

    if weights_df["criterion"].duplicated().any():
        raise ValueError("CSV bobot mengandung kriteria duplikat.")

    weights_df = weights_df.set_index("criterion")

    missing_criteria = set(ALL_CRITERIA) - set(weights_df.index)
    if missing_criteria:
        raise ValueError(
            f"Bobot tidak tersedia untuk kriteria: "
            f"{sorted(missing_criteria)}"
        )

    # Pertahankan urutan kriteria yang digunakan oleh pipeline.
    weights_df = weights_df.loc[list(ALL_CRITERIA)]

    w2 = _validate_weights(
        weights_df["type2_fahp_weight"].to_numpy(dtype=float),
        "Bobot FAHP Type-2",
    )

    # Type-1 hanya digunakan jika tersedia lengkap dan valid.
    type1_available = "type1_fahp_weight" in weights_df.columns

    if type1_available:
        raw_w1 = weights_df["type1_fahp_weight"].to_numpy(dtype=float)

        if np.isfinite(raw_w1).all():
            w1 = _validate_weights(
                raw_w1,
                "Bobot FAHP Type-1",
            )
            type1_source = "provided_in_csv"
            type1_placeholder = False
        elif np.isnan(raw_w1).all():
            w1 = np.full(len(ALL_CRITERIA), np.nan)
            type1_source = "unavailable"
            type1_placeholder = True
        else:
            raise ValueError(
                "Bobot Type-1 hanya terisi sebagian atau mengandung "
                "nilai tak hingga. Lengkapi seluruh bobot atau hapus "
                "kolom type1_fahp_weight jika memang tidak tersedia."
            )
    else:
        w1 = np.full(len(ALL_CRITERIA), np.nan)
        type1_source = "unavailable"
        type1_placeholder = True

    table3 = pd.DataFrame(
        {
            "criterion": ALL_CRITERIA,
            "type2_fahp_weight": w2,
            "type1_fahp_weight": w1,
            "type1_is_placeholder": type1_placeholder,
            "type1_source": type1_source,
            "weight_source": f"existing_csv:{source_path.name}",
        }
    )

    table3["type2_rank"] = _rank_or_nan(w2)
    table3["type1_rank"] = _rank_or_nan(w1)
    table3 = table3.sort_values(
        "type2_rank",
        kind="stable",
    )

    table3.to_csv(outdir / "table_03_fahp_weights.csv", index=False)

    # Nilai CR ini berasal dari manuskrip, bukan perhitungan ulang.
    table4 = pd.DataFrame(
        {
            "expert": [
                "MSME office",
                "MSME owner",
                "Batik association",
                "Academic",
                "Group matrix",
            ],
            "CI": [0.083, 0.091, 0.079, 0.095, 0.089],
            "CR": [0.052, 0.057, 0.050, 0.059, 0.056],
            "source": "reported_in_manuscript_not_recomputed",
            "verified_by_current_run": False,
        }
    )
    table4.to_csv(outdir / "table_04_expert_cr.csv", index=False)

    ranked = rank_criteria(w2, list(ALL_CRITERIA))
    (outdir / "fahp_ranked_criteria.txt").write_text(
        ",".join(ranked),
        encoding="utf-8",
    )

    if type1_placeholder:
        print(
            "PERINGATAN: bobot Type-1 tidak tersedia. Bobot Type-1 dan "
            "ranking-nya disimpan sebagai NaN; tidak ada perbandingan "
            "numerik Type-1 versus Type-2 yang dapat disimpulkan."
        )

    print("Mode: bobot siap pakai dari CSV.")
    print("Ranking FAHP Type-2:", ranked)
    print(f"Output: {outdir / 'table_03_fahp_weights.csv'}")
    print(f"Output: {outdir / 'table_04_expert_cr.csv'}")


def _from_example_matrices(outdir: Path) -> None:
    """Menghitung bobot dari matriks contoh sintetis."""
    expert_matrices = {
        name: example_expert_matrix(seed)
        for name, seed in EXPERT_SEEDS.items()
    }

    if not expert_matrices:
        raise ValueError("Tidak ada matriks contoh yang tersedia.")

    cr_rows = []

    # CR setiap matriks contoh.
    for name, matrix in expert_matrices.items():
        result = consistency_ratio(matrix)
        cr_rows.append(
            {
                "expert": name,
                "CI": round(result.ci, 3),
                "CR": round(result.cr, 3),
                "consistent": result.consistent,
                "source": "synthetic_example_matrix",
                "verified_by_current_run": True,
            }
        )

    group_matrix = aggregate_group_matrix(
        list(expert_matrices.values())
    )
    group_result = consistency_ratio(group_matrix)

    cr_rows.append(
        {
            "expert": "Group matrix",
            "CI": round(group_result.ci, 3),
            "CR": round(group_result.cr, 3),
            "consistent": group_result.consistent,
            "source": "aggregated_synthetic_example_matrices",
            "verified_by_current_run": True,
        }
    )

    pd.DataFrame(cr_rows).to_csv(
        outdir / "table_04_expert_cr.csv",
        index=False,
    )

    # Hitung bobot Type-2 dan Type-1 dari matriks contoh.
    w2 = _validate_weights(
        type2_fahp_weights(group_matrix),
        "Bobot FAHP Type-2 dari matriks contoh",
    )
    w1 = _validate_weights(
        conventional_fahp_weights(group_matrix),
        "Bobot FAHP konvensional dari matriks contoh",
    )

    if len(w2) != len(ALL_CRITERIA) or len(w1) != len(ALL_CRITERIA):
        raise ValueError(
            "Panjang vektor bobot tidak sama dengan jumlah kriteria."
        )

    table3 = pd.DataFrame(
        {
            "criterion": ALL_CRITERIA,
            "type2_fahp_weight": w2,
            "type1_fahp_weight": w1,
            "type1_is_placeholder": False,
            "type1_source": "computed_from_synthetic_example_matrix",
            "weight_source": "synthetic_example_matrices",
        }
    )

    table3["type2_rank"] = _rank_or_nan(w2)
    table3["type1_rank"] = _rank_or_nan(w1)
    table3 = table3.sort_values(
        "type2_rank",
        kind="stable",
    )

    table3.to_csv(outdir / "table_03_fahp_weights.csv", index=False)

    ranked = rank_criteria(w2, list(ALL_CRITERIA))
    (outdir / "fahp_ranked_criteria.txt").write_text(
        ",".join(ranked),
        encoding="utf-8",
    )

    print(
        "PERINGATAN: mode ini memakai matriks contoh sintetis. "
        "Hasilnya bukan bukti perhitungan dari matriks pakar penelitian."
    )
    print("Ranking FAHP Type-2 dari matriks contoh:", ranked)
    print(f"Output: {outdir / 'table_03_fahp_weights.csv'}")
    print(f"Output: {outdir / 'table_04_expert_cr.csv'}")


def main() -> None:
    _validate_criteria(list(ALL_CRITERIA))

    parser = argparse.ArgumentParser(
        description=(
            "Menghasilkan bobot FAHP Type-2 dan FAHP konvensional. "
            "Tanpa --weights-csv, matriks contoh sintetis digunakan."
        )
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
    )
    parser.add_argument(
        "--weights-csv",
        default=None,
        help=(
            "CSV dengan kolom criterion,type2_fahp_weight dan opsional "
            "type1_fahp_weight. Jika Type-1 tidak tersedia, bobot dan "
            "ranking Type-1 akan kosong, bukan disamakan dengan Type-2."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.weights_csv:
        _from_weights_csv(args.weights_csv, outdir)
    else:
        _from_example_matrices(outdir)


if __name__ == "__main__":
    main()
