# Cara menjalankan (skrip Python lewat terminal; bukan Streamlit atau notebook)

Ada dua cara. **Cara A untuk semua orang** (memakai data contoh anonim). **Cara B hanya untuk
penulis** (memakai data asli yang bersifat rahasia dan tidak dirilis).

## 1. Instal (sekali saja)

Pakai **Python 3.11** (versi yang dipakai untuk hasil naskah: 3.11.9). Python 3.12 ke atas TIDAK
disarankan karena `scikit-fuzzy 0.4.2` membutuhkan modul `imp` dan `distutils` yang sudah dihapus.

    python -m venv .venv
    .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
    pip install -r requirements-pinned.txt

- `requirements-pinned.txt`: versi persis yang dipakai di naskah. **Pakai ini untuk mereproduksi hasil.**
- `requirements.txt`: batas versi minimum, hanya untuk pemakaian umum; angka bisa bergeser sedikit.

Di Windows, taruh venv di path pendek (mis. `python -m venv C:\vb`) bila muncul "filename too long".

## 2. Cara A: data contoh (publik)

    python scripts/generate_all_tables.py --data data/example_data.csv --outdir outputs

Data contoh adalah data sintetis anonim (848 baris, 16 indikator) agar kode bisa dijalankan dari awal
sampai akhir. **Angkanya TIDAK sama dengan angka di naskah**; hanya alurnya yang sama. Pada data
contoh tabel yang memuat nama usaha tidak dihasilkan.

## 3. Cara B: data asli (hanya penulis)

    python scripts/convert_excel_to_csv.py --xlsx Data_Upload_Batik.xlsx
    python scripts/generate_all_tables.py --data data/batik_asli.csv --imputed data/batik_imputed_frozen.csv --outdir outputs --weights-csv params/paper_table3_weights.csv

Proses memakan sekitar 1 sampai 5 menit dan tidak mencetak apa pun saat tahap bootstrap; jangan
menekan Ctrl+C. Semua tabel (`table_03` sampai `table_15`, nomor file = nomor tabel naskah) dan
gambar (`figure_*.png`) muncul di folder `outputs/`.

Angka kunci yang harus muncul pada Cara B: K=5 dengan 7 fitur; ARI 0,9762 (SD 0,0340) dan NMI 0,9709;
SBS (A9) SC 0,5747 / DBI 0,9047 / CHI 510,0; statistik Friedman DBI 84,08 (Windows). Waktu
komputasi (Tabel 15) bergantung pada mesin.

## Catatan penting
- `--weights-csv` memakai bobot FAHP Type-2 pada Tabel 3 naskah. Tanpa opsi ini program memakai matriks
  pakar ACAK (placeholder), sehingga bobot dan fitur terpilih tidak sama dengan naskah.
- Matriks perbandingan berpasangan tiap pakar tidak dirilis. Bobot FAHP konvensional (Type-1) karena itu
  disamakan dengan bobot Type-2 pada kode ini, sehingga perbandingan "FAHP biasa vs Type-2" (Tabel 8, 9,
  10) tidak memberi bukti keunggulan Type-2. Hal ini juga dinyatakan di naskah (Bagian 4.10).
- Data asli tidak punya kolom C10 (konstan); kolom ditambahkan otomatis sebagai konstanta dan
  dikeluarkan dari subset acak serta SBS (15 indikator nonkonstan).
- `--imputed data/batik_imputed_frozen.csv` memakai hasil imputasi KNN yang dibekukan agar angka sama di
  semua komputer. Tanpa itu, banyak tetangga berjarak sama sehingga imputasi bisa memilih tetangga
  berbeda menurut urutan baris atau platform (SC K=5 bergeser antara 0,4593 dan 0,4609).
- TOPSIS (Tabel 13 dan 14) memakai matriks keputusan pada skala asli 1 sampai 4; clustering memakai data
  Min-Max. Tabel 14 dihitung per cluster (gangguan bobot dengan tanda acak, 100 ulangan, seed 42).
- Semua hyperparameter baseline ada di `params/baseline_params.yaml`.
