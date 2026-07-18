import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path resolution — works both locally and when run from any directory
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_FILE  = os.path.join(ROOT_DIR, "data", "processed", "complete_prediction_results.csv")
AGG_FILE   = os.path.join(ROOT_DIR, "data", "processed", "processed_agg.csv")
MODEL_FILE = os.path.join(ROOT_DIR, "models", "best_models.pkl")
IMAGE_FILE = os.path.join(ROOT_DIR, "reports", "02_modeling", "model_comparison.png")
YEAR_LIST = [2027, 2028, 2029, 2030, 2031]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Disaster Prediction 2027-2031",
    page_icon="🌏",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS & Fonts
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Plus Jakarta Sans', sans-serif; }
h1,h2,h3 { font-family: 'Outfit', sans-serif !important; font-weight: 700 !important; }
.header-box {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
    padding: 32px; border-radius: 20px; color: white; margin-bottom: 28px;
    box-shadow: 0 10px 25px -5px rgba(59,130,246,0.35);
}
.metric-card {
    background: #FFFFFF;
    padding: 20px 16px;
    border-radius: 16px;
    border: 1px solid #F1F5F9;
    text-align: center;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }
.metric-icon { font-size: 26px; margin-bottom: 8px; }
.metric-label {
    color: #64748B; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px; line-height: 1.3;
}
.metric-value {
    color: #0F172A; font-size: 20px; font-weight: 800;
    line-height: 1.2; word-break: break-word;
}
.trend-box { padding: 14px; border-radius: 12px; margin-top: 14px; border: 1px solid rgba(0,0,0,0.04); }
.info-box {
    background: #F8FAFC; border-left: 5px solid #6366F1;
    padding: 18px; border-radius: 12px; margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
  <h1 style="color:white;margin:0;font-size:32px;letter-spacing:-0.5px;">
    🌏 Global Natural Disaster Analysis & Prediction
  </h1>
  <p style="margin:8px 0 0 0;opacity:0.9;font-size:15px;font-weight:300;">
    Interactive dashboard for exploring trends and projections of natural disaster frequencies
    for the next 5 years (2027–2031) based on Linear Regression &amp; KNN Regression.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data & models
# ---------------------------------------------------------------------------
for path, name in [(DATA_FILE, "complete_prediction_results.csv"),
                   (MODEL_FILE, "best_models.pkl"),
                   (AGG_FILE,   "processed_agg.csv")]:
    if not os.path.exists(path):
        st.error(f"❌ File not found: `{path}`")
        st.info("💡 Please run the preprocessing and model training scripts locally first.")
        st.stop()

df_pred = pd.read_csv(DATA_FILE)
df_agg  = pd.read_csv(AGG_FILE)
with open(MODEL_FILE, "rb") as f:
    trained_models = pickle.load(f)

GRAN_COL = "Granularity (years)" if "Granularity (years)" in df_pred.columns else "Granularity_Best"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
country  = st.sidebar.selectbox("Select Country", sorted(df_pred["Country"].unique()))
disaster = st.sidebar.selectbox(
    "Select Disaster Type",
    sorted(df_pred[df_pred["Country"] == country]["Disaster Type"].unique())
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background:#F8FAFC;padding:14px;border-radius:10px;border:1px solid #E2E8F0;">
  <span style="font-size:11px;font-weight:700;color:#64748B;">STUDENT</span>
  <p style="margin:5px 0 2px 0;font-weight:700;color:#0F172A;font-size:13px;">Bungalunna Nashuha Camelia</p>
  <span style="font-size:12px;color:#64748B;">IT - Dian Nuswantoro University</span><br>
  <span style="font-size:11px;color:#94A3B8;">Machine Learning UAS 2025/2026</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Compute 5-year projection
# ---------------------------------------------------------------------------
key = (country, disaster)
if key not in trained_models:
    st.error(f"Model for combination {country} – {disaster} is not available.")
    st.stop()

info       = trained_models[key]
model      = info["model"]
g          = info["granularity"]
model_name = info["model_name"]

proj_5yr = {}
for year in YEAR_LIST:
    bin_t = (year // g) * g
    pred  = max(0, round(model.predict([[bin_t]])[0], 2))
    proj_5yr[year] = pred

row_df  = df_pred[(df_pred["Country"] == country) & (df_pred["Disaster Type"] == disaster)]
mae_lr  = round(float(row_df.iloc[0].get("MAE_LR",  0)), 3) if not row_df.empty else "-"
mae_knn = round(float(row_df.iloc[0].get("MAE_KNN", 0)), 3) if not row_df.empty else "-"

# ---------------------------------------------------------------------------
# Metric cards (4 columns)
# ---------------------------------------------------------------------------
st.subheader(f"📊 Dashboard: {country} — {disaster}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-icon">🏆</div>
        <div class="metric-label">Best Model</div>
        <div class="metric-value" style="color:#6366F1;font-size:20px;">{model_name}</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-icon">⏳</div>
        <div class="metric-label">Temporal Granularity</div>
        <div class="metric-value" style="color:#14B8A6;">{g} Years</div>
    </div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-icon">📉</div>
        <div class="metric-label">MAE Linear Regression</div>
        <div class="metric-value" style="color:#EC4899;">{mae_lr}</div>
    </div>''', unsafe_allow_html=True)
with c4:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-label">MAE KNN Regression</div>
        <div class="metric-value" style="color:#F97316;">{mae_knn}</div>
    </div>''', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart + Table: 5-year projection
# ---------------------------------------------------------------------------
st.subheader("📅 Time Series Projection (2027-2031)")
col_chart, col_tabel = st.columns([3, 2])

with col_chart:
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    hist = df_agg[(df_agg["Country"] == country) &
                  (df_agg["Disaster Type"] == disaster)].sort_values("Year")
    if not hist.empty:
        ax.plot(hist["Year"], hist["Disaster_Count"],
                color="#6366F1", linewidth=2.5, marker="o", markersize=5,
                markerfacecolor="#4F46E5", markeredgecolor="#FFFFFF",
                label="Historical (1900-2023)")
    year_proj = list(proj_5yr.keys())
    value_proj = list(proj_5yr.values())
    bars = ax.bar(year_proj, value_proj, color="#FB923C", alpha=0.9,
                  width=0.55, edgecolor="#EA580C", linewidth=1.2,
                  label="Model Projection (2027-2031)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, str(h),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#EA580C")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.set_xlabel("Year", fontsize=10, fontweight="bold", color="#475569")
    ax.set_ylabel("Event Count", fontsize=10, fontweight="bold", color="#475569")
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#E2E8F0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_tabel:
    st.markdown("**Projection Details Table:**")
    tabel = pd.DataFrame({
        "Year": year_proj,
        "Predicted Events": value_proj,
        "Average/Year": [round(v / g, 2) for v in value_proj]
    })
    st.dataframe(tabel, use_container_width=True, hide_index=True)
    diff = value_proj[-1] - value_proj[0]
    if diff > 0:
        tren_label, tren_color, tren_text = "Increasing", "#FEF3C7", "#92400E"
    elif diff < 0:
        tren_label, tren_color, tren_text = "Decreasing", "#DCFCE7", "#166534"
    else:
        tren_label, tren_color, tren_text = "Stable", "#F1F5F9", "#475569"
    st.markdown(
        f'<div class="trend-box" style="background:{tren_color};">'
        f'<span style="font-size:11px;font-weight:700;color:#64748B;">Trend Signal 2027-2031</span>'
        f'<p style="margin:4px 0 0 0;font-size:22px;font-weight:800;color:{tren_text};">{tren_label}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Model evaluation section
# ---------------------------------------------------------------------------
st.subheader("📈 Model Evaluation & Reliability")
col_img, col_desc = st.columns([3, 2])

with col_img:
    if os.path.exists(IMAGE_FILE):
        st.image(IMAGE_FILE,
                 caption="Boxplot MAE, RMSE, R2 — Linear Regression vs KNN",
                 use_container_width=True)
    else:
        st.warning("⚠️ Evaluation chart is not found. Please download `model_comparison.png` from Google Colab or run evaluation locally.")

with col_desc:
    lr_wins  = (df_pred["Best_Model"] == "Linear Regression").sum()
    knn_wins = (df_pred["Best_Model"] == "KNN Regression").sum()
    total    = len(df_pred)
    st.markdown(f"""
<div class="info-box">
  <b style="color:#4F46E5;">Model Win Statistics ({total} combinations):</b>
  <ul style="margin:8px 0 0 0;padding-left:18px;font-size:14px;color:#475569;line-height:1.7;">
    <li><b>Linear Regression</b>: {lr_wins} combinations ({lr_wins/total*100:.1f}%)</li>
    <li><b>KNN Regression</b>: {knn_wins} combinations ({knn_wins/total*100:.1f}%)</li>
  </ul>
</div>
<p style="font-size:13px;color:#64748B;line-height:1.5;">
  <b>Interpretation:</b> Linear Regression is generally superior for time-series extrapolation
  as it models the constant rate of change. KNN is constrained within the range of training inputs (1900-2023).
  Adaptive granularity (1/2/3/5 years) is selected per country-disaster pair based on the lowest MAE.
</p>
""", unsafe_allow_html=True)
