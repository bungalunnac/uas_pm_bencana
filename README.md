# 🌏 Global Natural Disaster Frequency Prediction

*Machine Learning UAS Project — Computer Science, Dian Nuswantoro University*

This project predicts natural disaster frequencies by country and disaster type up to 2027 using **Linear Regression** and **KNN Regression** models trained on historical EM-DAT data (1900–2023).

---

## 📂 Directory Structure

```
uas_pm/
├── data/
│   ├── raw/          ← Raw EM-DAT dataset
│   └── processed/    ← Preprocessed & prediction output CSV files
├── notebooks/        ← Jupyter Notebooks (EDA, Modeling, Interpretation)
├── src/              ← Modular Python scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── models/           ← Trained serialized models (best_models.pkl)
├── reports/          ← Generated evaluation plots & charts
├── app/
│   └── app.py        ← Streamlit dashboard application
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python -m src.data_preprocessing
```

Outputs: `data/processed/processed_agg.csv`

### 3. Train Models

```bash
python -m src.train_model
```

Outputs:
- `data/processed/hasil_prediksi_lengkap.csv`
- `models/best_models.pkl`

### 4. Evaluate & Generate Reports

```bash
python -m src.evaluate_model
```

Outputs: `reports/perbandingan_model.png`, `reports/feature_importance.png`

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app/app.py
```

---

## 📊 Dataset

- **Source**: EM-DAT — The International Disaster Database
- **Range**: 1900–2023, 178 countries, 10 natural disaster types
- **Valid combinations**: 488 (Country × Disaster Type with at least 5 data points)

---

## 🤖 Methodology

| Aspect | Detail |
|---|---|
| Granularity | Adaptive: 1, 2, 3, or 5-year bins (optimized based on lowest MAE) |
| Data Split | 70% Train / 30% Test (chronologically ordered split for time-series validation) |
| Models | Linear Regression vs KNN Regression |
| Model Selection | Selected per combination based on test-set MAE performance |
| Target Prediction Year | 2027 (dashboard projects 5-year window 2027–2031) |
