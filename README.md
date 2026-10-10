# Joint Feature and Cluster Optimization with Type Two FAHP Guided Feature Selection and TOPSIS for Robust Batik MSME Prioritization

Repositori reproduksibilitas untuk Paper ID **20265381** (IJIES / INASS).

Repositori ini berisi seluruh pipeline komputasi yang dipakai untuk menghasilkan semua angka,
tabel, dan figure dalam manuskrip: preprocessing, pembobotan kriteria FAHP Type-2, optimasi
gabungan fitur-cluster (grid 90-konfigurasi), efisiensi Pareto dan seleksi mean-rank, analisis
stabilitas bootstrap, perbandingan baseline (CLARA, FCM, IT2FCM, DBSCAN, weighted K-Means), uji
signifikansi statistik, analisis ablasi, validitas khusus-fuzzy, TOPSIS khusus per-cluster, dan
analisis sensitivitas TOPSIS.

> **Kerahasiaan data.** Data MSME Batik asli (848 MSME, 16 indikator) bersifat rahasia secara
> komersial dan tidak dapat dipublikasikan (lihat pernyataan Data Availability di manuskrip).
> Repositori ini menyertakan **dataset contoh sintetis yang sudah dianonimkan**
> (`data/example_data.csv`, dibuat oleh `scripts/generate_example_data.py`) yang memiliki skema
> dan rentang nilai yang masuk akal sama persis, sehingga seluruh pipeline bisa langsung
> dijalankan dari awal sampai akhir tanpa perlu data asli. Untuk mereproduksi angka persis seperti
> di paper, ganti `data/example_data.csv` dengan salinan resmi data asli dengan skema yang sama
> (lihat `data/schema.md`) lalu jalankan ulang `scripts/generate_all_tables.py`.

## Struktur repositori

```
batik-msme-fahp-topsis/
├── README.md
├── requirements.txt
├── data/
│   ├── schema.md                  # kamus data untuk C1–C16 dan aturan imputasi/penskalaan
│   └── example_data.csv           # contoh sintetis anonim (hasil generate, bukan data MSME asli)
├── params/
│   ├── baseline_params.yaml       # seluruh hyperparameter baseline, seed, versi library
│   └── fahp_pairwise_template.yaml# template matriks perbandingan berpasangan 4 pakar
├── src/batik_fahp_topsis/
│   ├── __init__.py
│   ├── preprocessing.py           # imputasi KNN, penskalaan Min-Max, audit kualitas data
│   ├── fahp.py                    # pembobotan kriteria FAHP Type-2 dan FAHP konvensional Type-1
│   ├── feature_selection.py       # subset bertingkat hasil FAHP, subset acak, SBS
│   ├── validity_metrics.py        # SC, DBI, CHI, PC, PE, XB, FOU
│   ├── clustering.py              # grid search gabungan K-Means
│   ├── baselines.py               # CLARA, FCM, IT2FCM, DBSCAN, weighted K-Means
│   ├── pareto.py                  # efisiensi Pareto + seleksi mean-rank berbobot sama
│   ├── bootstrap.py               # resampling bootstrap, stabilitas ARI/NMI
│   ├── statistical_tests.py       # Friedman, Wilcoxon dengan koreksi Holm, ukuran efek rank-biserial
│   ├── topsis.py                  # TOPSIS khusus per-cluster + sensitivitas / uji tekanan
│   └── reproducibility.py         # kontrol seed terpusat
├── scripts/
│   ├── generate_example_data.py   # membangun dataset contoh sintetis anonim
│   ├── run_preprocessing.py
│   ├── run_fahp_weights.py        # Tabel 3, Tabel 4
│   ├── run_joint_optimization.py  # grid 90-konfigurasi -> results_90_configs.csv
│   ├── run_pareto_and_ranking.py  # Tabel 6, Tabel 7, Figure 3
│   ├── run_bootstrap_stability.py # Tabel 7 (ARI/NMI + SD), Tabel 12
│   ├── run_baseline_comparison.py # Tabel 8
│   ├── run_statistical_tests.py   # Tabel 9
│   ├── run_ablation.py            # Tabel 10
│   ├── run_fuzzy_validity.py      # Tabel 11
│   ├── run_topsis_priorities.py   # Tabel 13
│   ├── run_topsis_sensitivity.py  # Tabel 14, Figure 6
│   ├── run_runtime_benchmark.py   # Tabel 15
│   └── generate_all_tables.py     # script utama: satu kali jalan otoritatif, dari awal sampai akhir
├── tests/
│   └── test_metrics.py
└── outputs/                       # dibuat saat runtime; setiap tabel manuskrip dalam bentuk CSV
```

## Cara mulai cepat

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Bangun dataset contoh anonim (lewati langkah ini dan ganti dengan data asli yang resmi
#    jika Anda sudah punya akses, dengan tetap mempertahankan skema/kolom yang sama).
python scripts/generate_example_data.py

# 2. Jalankan seluruh pipeline dari awal sampai akhir. Ini menghasilkan ulang setiap tabel
#    manuskrip langsung dari data/example_data.csv (atau data asli Anda) tanpa langkah transkripsi
#    manual, menjawab Reviewer 1 Komentar 7.
python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs/
```

Setiap `outputs/table_*.csv` berkorespondensi satu-ke-satu dengan sebuah tabel manuskrip;
`outputs/figure_*.png` mereproduksi figure yang sesuai. `outputs/results_90_configs.csv` adalah
satu-satunya file hasil otoritatif: semua tabel dan figure lain diturunkan dari file ini.

## Pemetaan ke manuskrip

| Item manuskrip | Script | File keluaran |
|---|---|---|
| Tabel 3, Tabel 4 (bobot FAHP Type-2, CR pakar) | `run_fahp_weights.py` | `table_03_fahp_weights.csv`, `table_04_expert_cr.csv` |
| Tabel 6, Tabel 7 (himpunan Pareto, mean rank, ARI/NMI bootstrap + SD) | `run_pareto_and_ranking.py`, `run_bootstrap_stability.py` | `table_06_leading_configs.csv`, `table_07_pareto_bootstrap.csv`, `table_07_bootstrap_per_repetition.csv` |
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

## Parameter baseline (Reviewer 2, Komentar 3)

Semua hyperparameter baseline, skema inisialisasi, aturan seed acak, dan versi library dicatat
di `params/baseline_params.yaml` dan dibaca langsung oleh `src/batik_fahp_topsis/baselines.py`,
sehingga tidak ada pengaturan baseline yang di-hard-code secara diam-diam di script analisis.

## Seed acak dan reproduksibilitas (Reviewer 1, Komentar 7)

`src/batik_fahp_topsis/reproducibility.py` memusatkan setiap seed acak yang dipakai di seluruh
pipeline (`random_state = 42` untuk grid full-data yang deterministik; seed yang berbeda dan
tercatat untuk setiap repetisi bootstrap). `scripts/generate_all_tables.py` adalah satu-satunya
titik masuk otoritatif: ia memanggil setiap tahap secara berurutan dari `results_90_configs.csv`
yang sama, sehingga status Pareto, rank, konfigurasi terpilih, statistik bootstrap, dan setiap
tabel manuskrip dihasilkan otomatis dari satu file hasil, bukan ditranskripsi manual antar tahap.

## Lisensi

Lisensi MIT (lihat `LICENSE`). Dataset contoh yang dianonimkan bersifat sintetis dan boleh
didistribusikan bebas; dataset MSME asli bersifat rahasia dan tidak disertakan di sini (lihat
pernyataan Data Availability di manuskrip).

## Catatan penomoran dan reproduksibilitas
- Nomor file `table_NN_*.csv` mengikuti nomor tabel pada naskah. Tabel 1, 2, dan 5 (studi terkait,
  nilai hilang, pengaturan eksperimen) tidak dihasilkan program; pengaturan eksperimen ada di
  `params/baseline_params.yaml`. Figure 1, 4, dan 5 tidak dihasilkan oleh skrip di repositori ini.
- SBS (Tabel 10, A9) memakai 15 indikator nonkonstan; C10 (varians nol) dikeluarkan, sama seperti
  subset acak (A8).
- IT2FCM memakai reduksi tipe sederhana (rata-rata membership bawah dan atas), bukan Karnik-Mendel
  penuh; CLARA memakai solver medoid ala PAM buatan sendiri. Keduanya dijelaskan di
  `params/baseline_params.yaml`.
- Statistik Friedman dan runtime dapat sedikit berbeda antar sistem operasi/mesin. Angka naskah
  berasal dari Windows, Python 3.11.9, dengan versi library di `requirements-pinned.txt`.
