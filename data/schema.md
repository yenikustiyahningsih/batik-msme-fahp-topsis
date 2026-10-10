# Skema data

`data/example_data.csv` (dan dataset asli yang resmi/berwenang jika diletakkan di path yang sama)
harus berisi satu baris per MSME dan kolom-kolom berikut.

| Kolom | Deskripsi | Tipe |
|---|---|---|
| `msme_id` | ID enterprise yang sudah dianonimkan | string |
| `C1` | Number of Consumers (Jumlah Konsumen) | numerik |
| `C2` | Consumer Complaints (Keluhan Konsumen) | numerik |
| `C3` | Partnerships (Kemitraan) | numerik |
| `C4` | Customer Rewards (Penghargaan Pelanggan) | numerik |
| `C5` | Employee Training (Pelatihan Karyawan) | numerik |
| `C6` | Product Branding (Branding Produk) | numerik |
| `C7` | Raw Material Price Increase (kriteria **cost**: makin tinggi makin buruk) | numerik |
| `C8` | Owner Training (Pelatihan Pemilik) | numerik |
| `C9` | Business License (Izin Usaha) | numerik |
| `C10` | Motif Variations (Variasi Motif) | numerik |
| `C11` | Offline Purchasing Rules (Aturan Pembelian Offline) | numerik |
| `C12` | IT-Certified Employees (Karyawan Bersertifikat IT) | numerik |
| `C13` | Owner Education (Pendidikan Pemilik) | numerik |
| `C14` | Marketplace | numerik |
| `C15` | Online Payment Facilities (Fasilitas Pembayaran Online) | numerik |
| `C16` | Own Management Information System (Sistem Informasi Manajemen Sendiri) | numerik |

Nilai yang hilang dikodekan sebagai sel kosong (menjadi `NaN` saat dibaca). `src/batik_fahp_topsis/preprocessing.py`
mengimputasinya dengan K-nearest neighbors (`k = 5`, TANPA penskalaan dulu — imputasi dilakukan
pada skala indikator mentah, persis seperti di manuskrip, Bagian 3.2), lalu menormalisasi Min-Max
setiap indikator ke rentang `[0, 1]` untuk semua metode berbasis jarak.

Seluruh enam belas indikator dipertahankan setelah preprocessing. Pemilihan fitur (feature
selection, mereduksi menjadi subset 7–16 indikator) adalah tahap pipeline yang terpisah dan
belakangan, didorong oleh ranking FAHP Type-2 (`src/batik_fahp_topsis/feature_selection.py`),
bukan bagian dari preprocessing.

`C7` adalah satu-satunya kriteria berarah cost; semua kriteria lain yang dipertahankan berarah
benefit. Ini dipakai oleh `src/batik_fahp_topsis/topsis.py` saat membangun solusi ideal dan
anti-ideal.
