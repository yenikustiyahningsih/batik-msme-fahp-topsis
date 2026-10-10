"""Tahap 2: pembobotan kriteria FAHP Type-2 (dan FAHP konvensional) -> Tabel 3, Tabel 4."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import EXPERT_SEEDS, example_expert_matrix
from batik_fahp_topsis.fahp import (
    aggregate_group_matrix,
    conventional_fahp_weights,
    consistency_ratio,
    footprint_of_uncertainty,
    rank_criteria,
    type2_fahp_weights,
)
from batik_fahp_topsis.preprocessing import ALL_CRITERIA


def _from_weights_csv(path, outdir):
    """Mode bobot-siap-pakai: memakai bobot Tabel 3 naskah langsung (bukan matriks pakar acak)."""
    import numpy as np

    w = pd.read_csv(path).set_index("criterion").loc[list(ALL_CRITERIA)]
    w2 = w["type2_fahp_weight"].to_numpy(dtype=float)
    w2 = w2 / w2.sum()
    if "type1_fahp_weight" in w.columns:
        w1 = w["type1_fahp_weight"].to_numpy(dtype=float)
        w1 = w1 / w1.sum()
        placeholder = False
    else:
        w1 = w2.copy()
        placeholder = True
        print("PERINGATAN: bobot FAHP Type-1 tidak tersedia; sementara disamakan dengan Type-2 "
              "(placeholder) sampai matriks pakar asli tersedia. Hasil ablasi/pembanding FAHP-vs-Type-2 "
              "BELUM bermakna.")
    table3 = pd.DataFrame({"criterion": ALL_CRITERIA, "type2_fahp_weight": w2, "type1_fahp_weight": w1})
    table3["type1_is_placeholder"] = placeholder
    table3["type2_rank"] = table3["type2_fahp_weight"].rank(ascending=False, method="min").astype(int)
    table3["type1_rank"] = table3["type1_fahp_weight"].rank(ascending=False, method="min").astype(int)
    table3 = table3.sort_values("type2_rank")
    table3.to_csv(outdir / "table_03_fahp_weights.csv", index=False)
    # Tabel 4 diambil dari nilai CR yang dilaporkan di naskah (bukan dihitung ulang).
    pd.DataFrame(
        {
            "expert": ["MSME office", "MSME owner", "Batik association", "Academic", "Group matrix"],
            "CI": [0.083, 0.091, 0.079, 0.095, 0.089],
            "CR": [0.052, 0.057, 0.050, 0.059, 0.056],
            "source": "dilaporkan di naskah (Tabel 4), bukan dihitung ulang",
        }
    ).to_csv(outdir / "table_04_expert_cr.csv", index=False)
    ranked = rank_criteria(w2, list(ALL_CRITERIA))
    with open(outdir / "fahp_ranked_criteria.txt", "w", encoding="utf-8") as f:
        f.write(",".join(ranked))
    print("Ranking FAHP Type-2 (dari Tabel 3 naskah):", ranked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--weights-csv",
        default=None,
        help="CSV (kolom: criterion,type2_fahp_weight[,type1_fahp_weight]) berisi bobot FAHP Type-2 "
        "yang sudah ada, mis. params/paper_table3_weights.csv (Tabel 3 naskah). Jika diberikan, "
        "matriks pakar sintetis TIDAK dipakai.",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.weights_csv:
        _from_weights_csv(args.weights_csv, outdir)
        return

    expert_matrices = {name: example_expert_matrix(seed) for name, seed in EXPERT_SEEDS.items()}

    # Tabel 4: consistency ratio per pakar, diperiksa sebelum fuzzifikasi.
    cr_rows = []
    for name, matrix in expert_matrices.items():
        r = consistency_ratio(matrix)
        cr_rows.append({"expert": name, "CI": round(r.ci, 3), "CR": round(r.cr, 3), "consistent": r.consistent})
    group_matrix = aggregate_group_matrix(list(expert_matrices.values()))
    r_group = consistency_ratio(group_matrix)
    cr_rows.append({"expert": "Group matrix", "CI": round(r_group.ci, 3), "CR": round(r_group.cr, 3), "consistent": r_group.consistent})
    table4 = pd.DataFrame(cr_rows)
    table4.to_csv(outdir / "table_04_expert_cr.csv", index=False)

    # Tabel 3: bobot dan rank FAHP Type-2 (bobot FAHP konvensional disertakan untuk pembanding).
    w_type2 = type2_fahp_weights(group_matrix)
    w_type1 = conventional_fahp_weights(group_matrix)

    table3 = pd.DataFrame(
        {
            "criterion": ALL_CRITERIA,
            "type2_fahp_weight": w_type2,
            "type1_fahp_weight": w_type1,
        }
    )
    table3["type2_rank"] = table3["type2_fahp_weight"].rank(ascending=False, method="min").astype(int)
    table3["type1_rank"] = table3["type1_fahp_weight"].rank(ascending=False, method="min").astype(int)
    table3 = table3.sort_values("type2_rank")
    table3.to_csv(outdir / "table_03_fahp_weights.csv", index=False)

    ranked = rank_criteria(w_type2, list(ALL_CRITERIA))
    with open(outdir / "fahp_ranked_criteria.txt", "w", encoding="utf-8") as f:
        f.write(",".join(ranked))

    print("Ranking FAHP Type-2 (bobot menurun):", ranked)
    print(f"Menulis {outdir / 'table_03_fahp_weights.csv'} dan {outdir / 'table_04_expert_cr.csv'}")


if __name__ == "__main__":
    main()
