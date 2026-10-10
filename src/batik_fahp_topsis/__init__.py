"""
batik_fahp_topsis
==================

Paket reproduksibilitas untuk "Joint Feature and Cluster Optimization with Type Two FAHP
Guided Feature Selection and TOPSIS for Robust Batik MSME Prioritization" (IJIES Paper
ID 20265381).

Sub-modul
---------
preprocessing        Imputasi KNN, penskalaan Min-Max, audit kualitas data (Bagian 3.2)
fahp                 Pembobotan kriteria FAHP Type-2 dan FAHP konvensional Type-1 (Bagian 3.3)
feature_selection    Subset bertingkat hasil ranking FAHP, subset acak, sequential backward selection
validity_metrics     SC, DBI, CHI, PC, PE, XB, FOU
clustering           Grid search gabungan fitur-cluster dengan K-Means (Bagian 3.4)
baselines            CLARA, FCM, IT2FCM, DBSCAN, weighted K-Means (Bagian 3.5)
pareto               Efisiensi Pareto + seleksi mean-rank berbobot sama (Bagian 3.4, Algoritma 1)
bootstrap            Resampling bootstrap, stabilitas ARI/NMI
statistical_tests    Uji Friedman, Wilcoxon signed-rank dengan koreksi Holm, ukuran efek rank-biserial
topsis               TOPSIS khusus per-cluster, analisis sensitivitas, uji tekanan bobot-sama
reproducibility      Kontrol seed acak terpusat
"""

__version__ = "1.0.0"
