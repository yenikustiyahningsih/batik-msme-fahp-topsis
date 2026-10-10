# Joint Feature and Cluster Optimization with Type-2 FAHP-Guided Feature Selection and TOPSIS for Robust Batik MSME Prioritization

[English](README.md) | Bahasa Indonesia

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23287436.svg)](https://doi.org/10.5281/zenodo.23287436)

Repositori reproduksibilitas untuk naskah **Paper ID 20265381** (IJIES / INASS).

Repositori ini berisi pipeline komputasi yang dipakai untuk menghasilkan tabel dan figure di
naskah: preprocessing, pembobotan kriteria FAHP Type-2, optimasi gabungan fitur-cluster (grid
90 konfigurasi), efisiensi Pareto dan seleksi mean-rank, analisis stabilitas bootstrap,
perbandingan baseline (CLARA, FCM, IT2FCM, weighted K-Means), uji signifikansi statistik, analisis
ablasi, validitas khusus-fuzzy, TOPSIS per-cluster, dan analisis sensitivitas TOPSIS.

> **Kerahasiaan data.** Data MSME Batik asli (848 MSME, 16 indikator) bersifat rahasia secara
> komersial dan **tidak** disertakan (lihat pernyataan Data Availability di naskah). Repositori ini
> menyediakan **dataset contoh sintetis anonim** (`data/example_data.csv`, dibuat oleh
> `scripts/generate_example_data.py`) dengan skema dan rentang nilai yang masuk akal, sehingga
> seluruh pipeline bisa dijalankan dari awal sampai akhir tanpa data asli. Hasil pada data contoh
> **tidak sama dengan angka di naskah**; hanya alurnya yang sama. Untuk mereproduksi angka naskah
> diperlukan salinan resmi data asli dengan skema yang sama (lihat `data/schema.md`).

## Cara mulai cepat

Pakai **Python 3.11** (hasil naskah dibuat dengan 3.11.9). Python 3.12 ke atas tidak didukung karena
`scikit-fuzzy 0.4.2` membutuhkan modul `imp` dan `distutils` yang sudah dihapus.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-pinned.txt

# Opsional: bangun ulang dataset contoh anonim
python scripts/generate_example_data.py

# Jalankan seluruh pipeline pada data contoh
python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs
```

- `requirements-pinned.txt`: versi persis yang dipakai di naskah. Pakai ini untuk mereproduksi hasil.
- `requirements.txt`: batas versi minimum untuk pemakaian umum; angka bisa bergeser sedikit.

Proses memakan beberapa menit dan hampir tidak mencetak apa pun saat tahap bootstrap; jangan
dihentikan. Setiap `outputs/table_*.csv` berkorespondensi dengan satu tabel naskah dan setiap
`outputs/figure_*.png` dengan satu figure. `outputs/results_90_configs.csv` adalah satu-satunya file
hasil otoritatif; semua tabel dan figure lain diturunkan dari file ini. Panduan menjalankan yang
lebih rinci ada di `CARA_MENJALANKAN.md`.

## Struktur repositori

```
batik-msme-fahp-topsis/
├── README.md / README.id.md
├── CARA_MENJALANKAN.md            # panduan menjalankan yang rinci
├── requirements.txt               # batas versi minimum
├── requirements-pinned.txt        # versi persis yang dipakai di naskah
├── data/
│   ├── schema.md                  # kamus data C1-C16, aturan imputasi dan penskalaan
│   └── example_data.csv           # contoh sintetis anonim (bukan data MSME asli)
├── params/
│   ├── baseline_params.yaml       # hyperparameter baseline, seed, versi library
│   ├── fahp_pairwise_template.yaml# template matriks perbandingan berpasangan 4 pakar
│   └── paper_table3_weights.csv   # bobot FAHP Type-2 pada Tabel 3
├── src/batik_fahp_topsis/
│   ├── preprocessing.py           # imputasi KNN, penskalaan Min-Max, audit kualitas data
│   ├── fahp.py                    # pembobotan FAHP Type-2 dan FAHP konvensional (Type-1)
│   ├── feature_selection.py       # subset hasil ranking FAHP, subset acak, SBS
│   ├── validity_metrics.py        # SC, DBI, CHI, PC, PE, XB, FOU
│   ├── clustering.py              # grid search gabungan K-Means
│   ├── baselines.py               # CLARA, FCM, IT2FCM, weighted K-Means (lihat catatan)
│   ├── pareto.py                  # efisiensi Pareto + seleksi mean-rank berbobot sama
│   ├── bootstrap.py               # resampling bootstrap, stabilitas ARI/NMI
│   ├── statistical_tests.py       # Friedman, Wilcoxon dengan koreksi Holm, ukuran efek rank-biserial
│   ├── topsis.py                  # TOPSIS per-cluster + sensitivitas / uji tekanan
│   └── reproducibility.py         # kontrol seed terpusat
├── scripts/                       # satu run_*.py per tahap analisis, plus generate_all_tables.py
├── tests/test_metrics.py
└── outputs/                       # dibuat saat runtime
```

## Pemetaan ke naskah

| Item naskah | Script | File keluaran |
|---|---|---|
| Tabel 3, 4 (bobot FAHP Type-2, CR pakar) | `run_fahp_weights.py` | `table_03_fahp_weights.csv`, `table_04_expert_cr.csv` |
| Tabel 6, 7 (himpunan Pareto, mean rank, ARI/NMI bootstrap + SD) | `run_pareto_and_ranking.py`, `run_bootstrap_stability.py` | `table_06_leading_configs.csv`, `table_07_pareto_bootstrap.csv`, `table_07_bootstrap_per_repetition.csv` |
| Figure 2 (heatmap skor komposit) | `run_joint_optimization.py` | `figure_02_composite_score.png` |
| Figure 3 (scatter trade-off) | `run_pareto_and_ranking.py` | `figure_03_tradeoff.png` |
| Tabel 8 (perbandingan baseline) | `run_baseline_comparison.py` | `table_08_baseline_comparison.csv` |
| Tabel 9 (Friedman / Wilcoxon-Holm) | `run_statistical_tests.py` | `table_09_statistical_tests.csv`, `table_09_holm_wilcoxon_{SC,DBI,CHI}.csv` |
| Tabel 10 (ablasi) | `run_ablation.py` | `table_10_ablation.csv` |
| Tabel 11 (validitas khusus-fuzzy) | `run_fuzzy_validity.py` | `table_11_fuzzy_validity.csv` |
| Tabel 12 (distribusi cluster akhir) | `run_bootstrap_stability.py` | `table_12_cluster_distribution.csv` |
| Tabel 13 (prioritas TOPSIS) | `run_topsis_priorities.py` | `table_13_topsis_priorities.csv` |
| Tabel 14, Figure 6 (sensitivitas TOPSIS) | `run_topsis_sensitivity.py` | `table_14_topsis_sensitivity.csv`, `figure_06_topsis_sensitivity.png` |
| Tabel 15 (runtime) | `run_runtime_benchmark.py` | `table_15_runtime.csv` |

Pada data contoh, tabel yang memuat nama usaha (misalnya Tabel 13) tidak dihasilkan.

## Parameter baseline

Semua hyperparameter baseline, skema inisialisasi, aturan seed acak, dan versi library dicatat di
`params/baseline_params.yaml` dan dibaca langsung oleh `src/batik_fahp_topsis/baselines.py`,
sehingga tidak ada pengaturan baseline yang di-hard-code di script analisis.

## Seed acak dan reproduksibilitas

`src/batik_fahp_topsis/reproducibility.py` memusatkan setiap seed acak (`random_state = 42` untuk
grid full-data yang deterministik; seed berbeda dan tercatat untuk setiap repetisi bootstrap).
`scripts/generate_all_tables.py` adalah satu-satunya titik masuk otoritatif: ia memanggil setiap
tahap secara berurutan dari `results_90_configs.csv` yang sama, sehingga status Pareto, rank,
konfigurasi terpilih, statistik bootstrap, dan setiap tabel naskah dihasilkan otomatis dari satu
file hasil, bukan ditranskripsi manual antar tahap.

## Catatan

- Penomoran: `table_NN_*.csv` mengikuti nomor tabel naskah. Tabel 1, 2, dan 5 (studi terkait, nilai
  hilang, pengaturan eksperimen) bukan keluaran program; pengaturan eksperimen ada di
  `params/baseline_params.yaml`. Figure 1, 4, dan 5 tidak dihasilkan oleh repositori ini.
- SBS (Tabel 10, A9) memakai 15 indikator nonkonstan; C10 (varians nol) dikeluarkan, sama seperti
  subset acak (A8).
- IT2FCM memakai reduksi tipe sederhana (rata-rata membership bawah dan atas), bukan Karnik-Mendel
  penuh; CLARA memakai solver medoid ala PAM buatan sendiri. Keduanya dijelaskan di
  `params/baseline_params.yaml`.
- `run_dbscan()` tetap ada di `baselines.py` hanya sebagai utilitas. DBSCAN sudah dihapus dari
  naskah dan tidak dipanggil pipeline.
- Matriks perbandingan berpasangan pakar yang asli tidak dirilis. Bobot FAHP konvensional (Type-1)
  karena itu disamakan dengan bobot Type-2 pada kode ini, sehingga perbandingan "FAHP biasa vs
  Type-2" (Tabel 8, 9, 10) tidak dengan sendirinya membuktikan keunggulan Type-2. Hal ini juga
  dinyatakan di naskah (Bagian 4.10).
- Statistik Friedman dan runtime dapat sedikit berbeda antar sistem operasi dan mesin. Angka naskah
  berasal dari Windows, Python 3.11.9, dengan versi library di `requirements-pinned.txt`.

## Sitasi

Jika memakai kode ini, mohon sitasi arsip Zenodo: https://doi.org/10.5281/zenodo.23287436

## Lisensi

Lisensi MIT (lihat `LICENSE`). Dataset contoh anonim bersifat sintetis dan boleh didistribusikan
bebas; dataset MSME asli bersifat rahasia dan tidak disertakan di sini.