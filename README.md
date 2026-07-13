# Analisis Tren dan Prediksi Frekuensi Bencana Alam
## Dataset EM-DAT (Emergency Events Database)

## Deskripsi Proyek
Proyek ini menganalisis tren frekuensi bencana alam secara global 
menggunakan dataset EM-DAT (1900-2023), dengan fokus pada prediksi 
jumlah kejadian bencana per negara dan per jenis bencana 
menggunakan pendekatan Machine Learning (Linear Regression).

## Dataset
- **Sumber:** EM-DAT - Emergency Events Database
- **Link Kaggle:** https://www.kaggle.com/datasets/mexwell/natural-disasters-emergency-events-database
- **Ukuran:** 10.431 baris × 13 kolom
- **Rentang Waktu:** 1900 - 2023

## Metodologi (CRISP-DM)
1. Business Understanding — Identifikasi masalah tren bencana global
2. Data Understanding — Eksplorasi karakteristik dataset EM-DAT
3. Data Preparation — Agregasi per negara, jenis bencana, dan tahun
4. Modeling — Linear Regression dengan evaluasi granularitas waktu terbaik
5. Evaluation — MAE sebagai metrik error utama
6. Deployment — (Rencana ke depan)

## Struktur Repository
├── notebooks/
│   └── pm_bencana.ipynb    # Notebook utama
├── data/
│   └── README.md           # Link ke sumber dataset
└── README.md

## Hasil Utama
- 488 kombinasi negara × jenis bencana berhasil diprediksi
- Rata-rata MAE: 0.141
- Granularitas terbaik: mayoritas per 1 tahun (459/488 kombinasi)
- Prediksi 2027: Banjir di Indonesia diperkirakan 2.66 kejadian/tahun

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (Linear Regression)
- Matplotlib, Seaborn
- Google Colab

## Mahasiswa
- Nama: [Nama kamu]
- NIM: [NIM kamu]
- Mata Kuliah: Pembelajaran Mesin
- Universitas Dian Nuswantoro
