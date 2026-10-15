# Proyek UTS: Klasifikasi Kondisi Ruangan

## Deskripsi
Proyek ini membangun pipeline Machine Learning untuk mengklasifikasikan kondisi ruangan berdasarkan suhu, kelembapan relatif, dan intensitas cahaya. Model baseline yang digunakan adalah Decision Tree.

## Dataset
File `dataset_kondisi_ruangan.csv` berisi 600 baris dataset dengan kolom:
- `id_data`: ID baris
- `suhu`: suhu dalam °C
- `kelembapan`: kelembapan relatif dalam %RH
- `cahaya`: intensitas cahaya dalam lux
- `kondisi_ruangan`: label kelas

Dataset sengaja mengandung beberapa nilai sensor yang kosong untuk menguji preprocessing.

## Definisi label operasional
Rentang target yang diasumsikan untuk proyek:
- Suhu: 22–28 °C
- Kelembapan: 40–65 %RH
- Cahaya: 300–700 lux

Label:
- `Nyaman`: semua fitur berada dalam rentang target.
- `Kurang Nyaman`: tepat satu fitur berada di luar rentang target.
- `Tidak Nyaman`: dua atau lebih fitur berada di luar rentang target.

Batas ini merupakan asumsi proyek, bukan standar universal kenyamanan ruangan.

## Instalasi
Jalankan di terminal atau sel Jupyter:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Cara menjalankan
Pastikan file CSV dan file Python berada dalam folder yang sama, lalu jalankan:

```bash
python klasifikasi_kondisi_ruangan.py
```

Di Jupyter Notebook, kode dalam file Python dapat ditempelkan atau dijalankan dengan:

```python
%run klasifikasi_kondisi_ruangan.py
```

## Pipeline
1. Membaca dan memeriksa dataset.
2. Memeriksa missing values dan nilai di luar rentang fisik yang diasumsikan.
3. Memvisualisasikan distribusi kelas dan korelasi.
4. Memisahkan data training dan testing (80:20, stratified).
5. Mengisi missing values menggunakan median dalam pipeline.
6. Melatih Decision Tree.
7. Mengevaluasi accuracy, precision, recall, F1-score, dan confusion matrix.
8. Memvisualisasikan pohon keputusan dan feature importance.
9. Melakukan prediksi contoh dan menyimpan model.

Decision Tree tidak memerlukan standardisasi fitur untuk bekerja dengan baik, sehingga normalisasi tidak diterapkan. Imputer hanya dipelajari dari data training melalui pipeline untuk mengurangi risiko data leakage.

## Batasan
Dataset ini adalah data simulasi dan label dibuat berdasarkan aturan ambang di atas. Karena itu, skor evaluasi mengukur seberapa baik model mempelajari aturan pada data simulasi, bukan bukti bahwa sistem sudah akurat di ruangan nyata. Untuk penggunaan nyata, kumpulkan data dari sensor, dokumentasikan waktu/lokasi pengukuran, dan tetapkan label berdasarkan protokol yang dapat dipertanggungjawabkan.

## Struktur repository yang disarankan
```text
klasifikasi-kondisi-ruangan/
├── dataset_simulasi_kondisi_ruangan.csv
├── klasifikasi_kondisi_ruangan.py
├── README.md
└── model_kondisi_ruangan.joblib  # dibuat setelah program dijalankan
```
