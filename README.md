# 🌏 Global Natural Disaster Analysis & Prediction

Final Exam (UAS) Capstone Project — **Machine Learning** Course
Informatics Engineering, Universitas Dian Nuswantoro, Semarang (2025/2026)

Forecasting global natural disaster frequency for 2027–2031 using **Linear Regression** and **KNN Regression**, based on 124 years of historical EM-DAT data (1900–2023), packaged into an interactive Streamlit dashboard.

🔗 **Live Demo:** (https://15611-uaspmbencana.streamlit.app/)
📄 **Full Technical Report:** [`reports/Laporan_UAS_Pembelajaran_Mesin.pdf`](./reports/)

---

## 📌 Project Summary

| | |
|---|---|
| **Author** | Bungalunna Nashuha Camelia — A11.2024.15611 |
| **Course** | Machine Learning (Even Semester UAS 2025/2026) |
| **Lecturer** | Abu Salam, M.Kom |
| **Dataset** | [EM-DAT — Emergency Events Database](https://www.kaggle.com/datasets/mexwell/natural-disasters-emergency-events-database) (via Kaggle, by Mexwell) |
| **Problem Type** | Supervised Learning — Regression (not classification) |
| **Prediction Target** | `Disaster_Count` — number of disaster events per Country × Disaster Type combination |
| **Algorithms** | Linear Regression vs KNN Regression |

## 🎯 Problem Statement

Claims that global natural disaster frequency keeps rising are common in the media, but aren't always backed by quantitative analysis. This project tests whether 124 years of historical EM-DAT data (1900–2023) can be used to measure long-term disaster frequency trends in a rigorous, data-driven way, while also building a forecasting model for disaster counts in 2027–2031 for each Country × Disaster Type combination.

**Success metrics:**
- MAE < 0.5 events per prediction period for the best model
- Best model outperforms a naive baseline (historical average)
- Trend analysis can confirm or refute the hypothesis that disaster frequency is increasing

## 🔍 Methodology Overview

The project uses **two complementary evaluation schemes**:

1. **Per-Combination Scheme** — 488 individual time series (out of 1,035 available Country × Disaster Type combinations, filtered to require at least 5 unique years of data), each with adaptive temporal granularity (1/2/3/5 years) and a 70/30 train/test split.
2. **Global Time Series Scheme** — all countries combined per year (124 data points), validated more rigorously with a Train/Validation/Test split (70/15/15) and KNN hyperparameter tuning via GridSearchCV.

**Key results:**

| Scheme | Best Model | Detail |
|---|---|---|
| Per-Combination (488 combinations) | Linear Regression | Wins in 77% of combinations (376/488); average MAE 0.141 vs KNN's 0.135 |
| Global Time Series (Train/Val/Test) | KNN Regression (tuned) | Performs better on the 2005–2023 test set, though both models show negative R² due to difficulty extrapolating recent trend acceleration |

**2027 prediction for Indonesia:** Flood (2.66 events) > Storm (1.54) > Earthquake (1.20) — Flood remains the top mitigation priority.

Full details on EDA, feature importance (trend coefficients), residual analysis, and business interpretation are available in the technical report and notebooks.

## 📊 Dashboard

The Streamlit dashboard displays, for the Country × Disaster Type combination selected by the user:
- Best model & temporal granularity used
- Event count projection for 2027–2031 (chart + table)
- Linear Regression vs KNN performance comparison (488 combinations)
- Model interpretation insights

> **Note:** the current dashboard is a single-page layout with sidebar filters (Select Country & Select Disaster Type); it does not yet include separate EDA and documentation pages — see [Future Improvements](#-future-improvements) below.

## 🗂️ Repository Structure

```
uas_pm/
├── app/
│   └── app.py                        # Main Streamlit application
├── data/
│   ├── raw/
│   │   └── EmergencyEventsDatabase...csv
│   └── processed/
│       ├── processed_agg.csv
│       ├── processed_indo_flood.csv
│       └── complete_prediction_results.csv
├── models/
│   └── best_models.pkl                # 488 best models + granularity per combination
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_modeling.ipynb              # Preprocessing, training, tuning
│   └── 03_interpretation.ipynb        # Evaluation, residuals, feature importance
├── reports/
│   ├── 01_eda/                        # EDA output visualizations
│   ├── 02_modeling/                   # Modeling output visualizations
│   └── 03_interpretation/             # Model interpretation visualizations
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone (https://github.com/bungalunnac/global-disaster-prediction.git)
cd uas_pm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app/app.py
```

Run the notebooks in order — `01_eda.ipynb` → `02_modeling.ipynb` → `03_interpretation.ipynb` — to reproduce the full pipeline from raw data to the saved model (`best_models.pkl`).

## 🧰 Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit

## 📈 Future Improvements

- Add predictors beyond time (climate indicators, population, GDP per capita)
- Explore additional non-linear models (Polynomial Regression, Random Forest Regressor)
- Add separate EDA and Documentation pages to the dashboard
- Migrate to a permanent Streamlit Community Cloud deployment
- Set up periodic retraining as new EM-DAT data is released

## 📚 References

- CRED — Centre for Research on the Epidemiology of Disasters. [EM-DAT: The International Disaster Database](https://www.emdat.be/)
- Mexwell. (2023). [Natural Disasters – Emergency Events Database (EM-DAT)](https://www.kaggle.com/datasets/mexwell/natural-disasters-emergency-events-database). Kaggle.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.

---

*This project was created to fulfill the Even Semester 2025/2026 Final Exam (UAS) requirement for the Machine Learning course, Universitas Dian Nuswantoro.*
